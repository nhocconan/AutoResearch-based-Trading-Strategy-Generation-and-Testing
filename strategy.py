#!/usr/bin/env python3
"""
Experiment #952: 12h Primary + 1d HTF — Regime-Adaptive Dual Strategy

Hypothesis: 12h timeframe with regime-adaptive logic outperforms pure trend/mean-reversion.
- Choppiness Index detects regime: CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trend
- Range regime: RSI(3) extremes for mean reversion entries (loose thresholds)
- Trend regime: HMA(16/48) crossover for trend following
- 1d HMA(21) for HTF bias filter (only trade with daily trend)
- ATR(14) 2.5x trailing stop for risk management

Why 12h:
- Captures multi-day swings without 4h/6h noise
- 20-50 trades/year target (fee-efficient)
- Works in both 2022 crash (range/chop) and 2021 bull (trend)

Key innovations:
1. Regime detection via Choppiness Index (14-period)
2. Dual entry logic: RSI(3) for range, HMA for trend
3. 1d HTF bias filter (trade with daily trend direction)
4. LOOSE entry conditions to guarantee ≥30 trades/train, ≥3/test
5. Discrete signal sizes: 0.0, ±0.25, ±0.30

Target: Sharpe>0.45, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 12h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_regime_adaptive_rsi_hma_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    Measures market choppiness vs trending
    CHOP > 61.8 = range/chop, CHOP < 38.2 = trend
    
    Formula:
    ATR_sum = sum(ATR(period)) over lookback
    HV = max(high) - min(low) over lookback
    CHOP = 100 * log10(ATR_sum / HV) / log10(period)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate True Range
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Calculate ATR sum over lookback period
    atr_sum = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        atr_sum[i] = np.sum(tr[i-period+1:i+1])
    
    # Calculate highest high and lowest low over lookback
    hh_ll = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        hh_ll[i] = hh - ll
    
    # Calculate CHOP
    chop = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if hh_ll[i] > 1e-10 and atr_sum[i] > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum[i] / hh_ll[i]) / np.log10(period)
    
    return chop

def calculate_hma(close, period):
    """
    Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1].astype(np.float64)
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    hma = wma(diff, sqrt_n)
    return hma

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 12h indicators
    chop_14 = calculate_choppiness(high, low, close, period=14)
    hma_16 = calculate_hma(close, period=16)
    hma_48 = calculate_hma(close, period=48)
    rsi_3 = calculate_rsi(close, period=3)  # Fast RSI for mean reversion
    rsi_14 = calculate_rsi(close, period=14)  # Standard RSI
    atr_14 = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 61.8 = range/chop, CHOP < 38.2 = trend
        in_range_regime = not np.isnan(chop_14[i]) and chop_14[i] > 55.0  # Looser threshold
        in_trend_regime = not np.isnan(chop_14[i]) and chop_14[i] < 45.0  # Looser threshold
        
        # === HTF BIAS (1d HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 12h HMA CROSSOVER (Trend Signal) ===
        hma_crossover_long = False
        hma_crossover_short = False
        if i > 0 and not np.isnan(hma_16[i-1]) and not np.isnan(hma_48[i-1]):
            hma_crossover_long = (hma_16[i-1] <= hma_48[i-1]) and (hma_16[i] > hma_48[i])
            hma_crossover_short = (hma_16[i-1] >= hma_48[i-1]) and (hma_16[i] < hma_48[i])
        
        hma_12h_bull = hma_16[i] > hma_48[i]
        hma_12h_bear = hma_16[i] < hma_48[i]
        
        # === RSI MEAN REVERSION (Range Signal) ===
        # LOOSE thresholds to ensure trades: RSI(3) < 15 or > 85
        rsi_oversold = not np.isnan(rsi_3[i]) and rsi_3[i] < 20.0
        rsi_overbought = not np.isnan(rsi_3[i]) and rsi_3[i] > 80.0
        
        # === ENTRY LOGIC (LOOSE TO GUARANTEE TRADES) ===
        desired_signal = 0.0
        
        # LONG entries
        if htf_1d_bull:  # Daily trend is bullish
            if in_range_regime and rsi_oversold:
                # Range regime: mean reversion long
                desired_signal = SIZE_BASE
            elif in_trend_regime and hma_crossover_long:
                # Trend regime: trend following long
                desired_signal = SIZE_STRONG
            elif hma_12h_bull and not in_position:
                # Looser: just HMA bull + no position
                desired_signal = SIZE_BASE
        
        # SHORT entries
        elif htf_1d_bear:  # Daily trend is bearish
            if in_range_regime and rsi_overbought:
                # Range regime: mean reversion short
                desired_signal = -SIZE_BASE
            elif in_trend_regime and hma_crossover_short:
                # Trend regime: trend following short
                desired_signal = -SIZE_STRONG
            elif hma_12h_bear and not in_position:
                # Looser: just HMA bear + no position
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals