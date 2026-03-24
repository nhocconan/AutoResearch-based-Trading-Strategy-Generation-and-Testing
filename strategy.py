#!/usr/bin/env python3
"""
Experiment #948: 4h Primary + 12h/1d HTF — Fisher Transform + HMA Trend + ADX Regime

Hypothesis: Fisher Transform excels at catching reversals in bear/range markets (2025 test period).
Combined with HMA trend filter and ADX regime detection, this should work across all market conditions.

Key innovations:
1. 12h HMA(21) for primary trend direction (proven MTF edge)
2. 4h Fisher Transform(9) for precise entry timing at extremes
3. ADX(14) regime filter: ADX>25=trend follow, ADX<20=mean revert
4. Dual-mode entries: trend continuation OR mean reversion based on regime
5. ATR(14) 2.5x trailing stop for risk management
6. LOOSE entry conditions to guarantee ≥10 trades/train, ≥3/test

Why Fisher Transform:
- Converts price to near-Gaussian distribution
- Sharp turning points at extremes (±1.5 to ±2.0)
- Proven in bear markets where EMA/HMA whipsaw
- Works on BTC/ETH/SOL equally (no SOL bias)

Entry conditions (LOOSE to guarantee trades):
- TREND MODE (ADX>25): Fisher crosses -1.5 up + HMA bull = LONG
- TREND MODE (ADX>25): Fisher crosses +1.5 down + HMA bear = SHORT
- RANGE MODE (ADX<20): Fisher < -2.0 = LONG, Fisher > +2.0 = SHORT
- TRANSITION (ADX 20-25): half size or flat

Target: Sharpe>0.45, trades>=20 train, trades>=5 test, DD>-40%
Timeframe: 4h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_hma_adx_regime_12h1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_fisher(close, period=9):
    """
    Ehlers Fisher Transform
    Converts price to Gaussian-like distribution for clearer turning points
    
    Steps:
    1. Normalize price to -1 to +1 range
    2. Apply Fisher transform: 0.5 * ln((1+x)/(1-x))
    3. Smooth with EMA
    """
    n = len(close)
    if n < period + 10:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_signal = np.full(n, np.nan, dtype=np.float64)
    
    # Find highest high and lowest low over lookback
    for i in range(period, n):
        hh = np.max(close[i-period+1:i+1])
        ll = np.min(close[i-period+1:i+1])
        
        if hh > ll:
            # Normalize to -1 to +1 (with 0.999 clamp to avoid division by zero)
            x = 0.999 * 2.0 * ((close[i] - ll) / (hh - ll)) - 1.0
            
            # Fisher transform
            if abs(x) < 0.999:
                fisher_val = 0.5 * np.log((1.0 + x) / (1.0 - x + 1e-10))
            else:
                fisher_val = np.sign(x) * 3.0  # Clamp extreme values
            
            fisher[i] = fisher_val
        else:
            fisher[i] = fisher[i-1] if i > 0 else 0.0
    
    # Smooth Fisher with EMA(3) for signal line
    valid_mask = ~np.isnan(fisher)
    if np.any(valid_mask):
        fisher_series = pd.Series(fisher)
        fisher_smooth = fisher_series.ewm(span=3, min_periods=3, adjust=False).mean().values
        fisher_signal[:] = fisher_smooth
        fisher[:] = fisher_smooth  # Use smoothed for both
    
    return fisher, fisher_signal

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

def calculate_adx(high, low, close, period=14):
    """Average Directional Index"""
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0.0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0.0
        
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth with Wilder's method (EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Avoid division by zero
    plus_di = np.where(atr > 1e-10, 100.0 * plus_di / atr, 0.0)
    minus_di = np.where(atr > 1e-10, 100.0 * minus_di / atr, 0.0)
    
    dx = np.zeros(n, dtype=np.float64)
    for i in range(n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 4h indicators
    fisher, fisher_signal = calculate_fisher(close, period=9)
    adx_14 = calculate_adx(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    SIZE_HALF = 0.15
    
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
        
        if np.isnan(fisher[i]) or np.isnan(adx_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS (12h + 1d HMA) ===
        htf_12h_bull = close[i] > hma_12h_aligned[i]
        htf_12h_bear = close[i] < hma_12h_aligned[i]
        
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Strong bias when both agree
        htf_bull = htf_12h_bull and htf_1d_bull
        htf_bear = htf_12h_bear and htf_1d_bear
        
        # === ADX REGIME DETECTION ===
        adx_val = adx_14[i]
        regime_trend = adx_val > 25.0  # Strong trend
        regime_range = adx_val < 20.0  # Range/chop
        regime_transition = not regime_trend and not regime_range  # ADX 20-25
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_up = False
        fisher_cross_down = False
        
        if i > 0 and not np.isnan(fisher[i-1]) and not np.isnan(fisher_signal[i-1]):
            # Fisher crosses above signal line
            fisher_cross_up = (fisher[i-1] <= fisher_signal[i-1]) and (fisher[i] > fisher_signal[i])
            # Fisher crosses below signal line
            fisher_cross_down = (fisher[i-1] >= fisher_signal[i-1]) and (fisher[i] < fisher_signal[i])
        
        # Fisher at extremes
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        fisher_extreme_low = fisher[i] < -2.0
        fisher_extreme_high = fisher[i] > 2.0
        
        # === ENTRY LOGIC (LOOSE TO GUARANTEE TRADES) ===
        desired_signal = 0.0
        
        # TREND MODE (ADX > 25): Follow HTF trend with Fisher timing
        if regime_trend:
            # LONG: HTF bull + Fisher crossover up or oversold
            if htf_bull:
                if fisher_cross_up or fisher_oversold:
                    desired_signal = SIZE_STRONG
                elif fisher[i] > fisher_signal[i]:  # Fisher above signal
                    desired_signal = SIZE_BASE
            
            # SHORT: HTF bear + Fisher crossover down or overbought
            elif htf_bear:
                if fisher_cross_down or fisher_overbought:
                    desired_signal = -SIZE_STRONG
                elif fisher[i] < fisher_signal[i]:  # Fisher below signal
                    desired_signal = -SIZE_BASE
        
        # RANGE MODE (ADX < 20): Mean reversion at Fisher extremes
        elif regime_range:
            if fisher_extreme_low:
                desired_signal = SIZE_BASE
            elif fisher_extreme_high:
                desired_signal = -SIZE_BASE
            elif fisher_oversold and htf_bull:
                desired_signal = SIZE_BASE
            elif fisher_overbought and htf_bear:
                desired_signal = -SIZE_BASE
        
        # TRANSITION MODE (ADX 20-25): Half size or flat
        elif regime_transition:
            if htf_bull and fisher_oversold:
                desired_signal = SIZE_HALF
            elif htf_bear and fisher_overbought:
                desired_signal = -SIZE_HALF
        
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
        elif desired_signal >= SIZE_HALF * 0.9:
            final_signal = SIZE_HALF
        elif desired_signal <= -SIZE_HALF * 0.9:
            final_signal = -SIZE_HALF
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