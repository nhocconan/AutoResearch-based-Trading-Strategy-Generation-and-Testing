#!/usr/bin/env python3
"""
Experiment #1277: 1d Primary + 1w HTF — Dual Regime RSI + HMA Trend

Hypothesis: Recent 1d strategies (#1267, #1273) failed with Sharpe=0.000 = ZERO TRADES.
Entry conditions were TOO STRICT with multiple confluence requirements.

This strategy uses:
1. CHOPPINESS INDEX for regime (wider thresholds: >55=range, <40=trend)
2. RSI(14) for entries with LOOSE thresholds (30-70 vs extreme 10-90)
3. 1w HMA(21) for macro trend bias (simple, not strict filter)
4. ATR(14) * 2.5 trailing stoploss

Key changes from failed 1d attempts:
- Remove strict ADX requirement (was blocking signals)
- Wider RSI entry bands (30-70 vs 20-80)
- Choppiness thresholds more permissive (40-55 buffer zone)
- Macro trend is BIAS not hard filter (can enter against it in range regime)
- BASE_SIZE = 0.30 for controlled drawdown

Target: Sharpe > 0.612, trades >= 40 train (10/year), >= 6 test
Timeframe: 1d
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_chop_rsi_regime_1w_hma_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[:period] = np.nan
    
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - regime detection
    CHOP > 61.8 = ranging (mean revert)
    CHOP < 38.2 = trending (trend follow)
    Using wider thresholds: >55=range, <40=trend
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        tr_sum = np.sum(tr[i-period+1:i+1])
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        if hh > ll and tr_sum > 0:
            chop[i] = 100.0 * np.log10(tr_sum / (hh - ll)) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(close, period=200):
    """Simple Moving Average"""
    n = len(close)
    sma = np.full(n, np.nan)
    
    if n < period:
        return sma
    
    for i in range(period - 1, n):
        sma[i] = np.mean(close[i-period+1:i+1])
    
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for macro trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    rsi = calculate_rsi(close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    sma200 = calculate_sma(close, period=200)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(rsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(sma200[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        # Wider thresholds to ensure signals trigger
        in_range = chop[i] > 55.0  # Ranging market
        in_trend = chop[i] < 40.0  # Trending market
        # 40-55 is buffer zone - use trend logic
        
        # === MACRO TREND BIAS (1w HMA) ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === LONG-TERM FILTER (SMA200) ===
        above_sma200 = close[i] > sma200[i]
        below_sma200 = close[i] < sma200[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # TRENDING REGIME: Follow macro trend on RSI pullback
        if in_trend:
            # Long: Macro bull + RSI pullback (not oversold, just dipping)
            if macro_bull and rsi[i] < 55.0 and rsi[i] > 35.0:
                desired_signal = BASE_SIZE
            # Short: Macro bear + RSI rally (not overbought, just rising)
            elif macro_bear and rsi[i] > 45.0 and rsi[i] < 65.0:
                desired_signal = -BASE_SIZE
        
        # RANGING REGIME: Mean revert at RSI extremes
        elif in_range:
            # Long: RSI oversold (loose threshold)
            if rsi[i] < 40.0:
                desired_signal = BASE_SIZE
            # Short: RSI overbought (loose threshold)
            elif rsi[i] > 60.0:
                desired_signal = -BASE_SIZE
        
        # BUFFER ZONE: Use ranging logic (more trades)
        else:
            # Long: RSI oversold + above SMA200 bias
            if rsi[i] < 45.0 and above_sma200:
                desired_signal = BASE_SIZE
            # Short: RSI overbought + below SMA200 bias
            elif rsi[i] > 55.0 and below_sma200:
                desired_signal = -BASE_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === OUTPUT SIGNAL ===
        final_signal = desired_signal
        
        # === DISCRETIZE SIGNAL VALUES ===
        if final_signal > 0.1:
            final_signal = BASE_SIZE
        elif final_signal < -0.1:
            final_signal = -BASE_SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals