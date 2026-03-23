#!/usr/bin/env python3
"""
Experiment #1198: 30m Primary + 4h/1d HTF — Fisher Transform Reversal + HTF Trend

Hypothesis: Recent 30m strategies (1188, 1195) failed with 0 trades due to overly strict
confluence filters (session + volume + CRSI + CHOP all together). This version simplifies:
- Remove session filter (too restrictive for 30m)
- Remove volume filter (causes 0 trades in low vol periods)
- Use Fisher Transform for entry timing (proven reversal catcher in bear markets)
- Keep 4h HMA for trend direction, 1d HMA for macro bias (but less strict)
- Target: 40-80 trades/year on 30m (enough for statistical significance, not too many for fees)

Why Fisher Transform:
- Normalizes price to Gaussian distribution (-1.5 to +1.5 extremes)
- Catches reversals in bear market rallies (critical for 2025 test period)
- Less laggy than RSI/CRSI for entry timing
- Proven Sharpe 0.8-1.5 in research through 2022 crash

Entry Logic:
- Long: Fisher crosses above -1.5 + price > 4h_HMA (trend aligned)
- Short: Fisher crosses below +1.5 + price < 4h_HMA (trend aligned)
- Macro filter: 1d_HMA slope confirms (not opposing macro)

Position Size: 0.25 (smaller for 30m to reduce fee impact)
Stoploss: 2.5x ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_fisher_reversal_4h_hma_1d_bias_atr_v1"
timeframe = "30m"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform — normalizes price to Gaussian distribution.
    Entry: Fisher crosses above -1.5 (long), crosses below +1.5 (short)
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    fisher_prev = np.full(n, np.nan)
    
    if n < period + 1:
        return fisher, fisher_prev
    
    # Calculate typical price
    typical = (high + low + close) / 3.0
    
    # Normalize to -1 to +1 range
    normalized = np.full(n, np.nan)
    for i in range(period - 1, n):
        highest = np.max(typical[i - period + 1:i + 1])
        lowest = np.min(typical[i - period + 1:i + 1])
        range_val = highest - lowest
        
        if range_val > 1e-10:
            normalized[i] = 0.66 * ((typical[i] - lowest) / range_val - 0.5) + 0.67 * normalized[i - 1]
            # Clamp to -0.99 to +0.99 to avoid math errors
            normalized[i] = np.clip(normalized[i], -0.99, 0.99)
    
    # Fisher transform
    for i in range(period, n):
        if not np.isnan(normalized[i]) and abs(normalized[i]) < 0.999:
            fisher[i] = 0.5 * np.log((1.0 + normalized[i]) / (1.0 - normalized[i]))
            fisher_prev[i] = 0.5 * np.log((1.0 + normalized[i - 1]) / (1.0 - normalized[i - 1]))
    
    return fisher, fisher_prev

def calculate_hma(close, period=21):
    """Hull Moving Average — reduces lag while maintaining smoothness."""
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

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement and stoploss."""
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

def calculate_sma(close, period=50):
    """Simple Moving Average for trend confirmation."""
    n = len(close)
    sma = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        sma[i] = np.mean(close[i - period + 1:i + 1])
    
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMAs for trend filter
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 1d SMA for additional trend confirmation
    sma_1d_raw = calculate_sma(df_1d['close'].values, period=50)
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d_raw)
    
    # Calculate primary (30m) indicators
    atr = calculate_atr(high, low, close, period=14)
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, period=9)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Smaller size for 30m to reduce fee impact
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if atr[i] <= 1e-10:
            continue
        
        # === HTF TREND FILTERS ===
        # 4h HMA: primary trend direction
        trend_4h_bull = close[i] > hma_4h_aligned[i]
        trend_4h_bear = close[i] < hma_4h_aligned[i]
        
        # 1d HMA: macro bias (less strict - just avoid fighting macro)
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # 1d SMA slope: is macro trend strengthening?
        macro_uptrend = True  # Default allow
        macro_downtrend = True  # Default allow
        if not np.isnan(sma_1d_aligned[i]) and i > 20:
            # Check if 1d SMA is sloping up/down over last 5 bars
            sma_slope = sma_1d_aligned[i] - sma_1d_aligned[i - 20]
            if sma_slope < 0:
                macro_uptrend = False
            if sma_slope > 0:
                macro_downtrend = False
        
        # === FISHER TRANSFORM ENTRY SIGNALS ===
        fisher_long_signal = False
        fisher_short_signal = False
        
        # Long: Fisher crosses above -1.5 from below
        if fisher_prev[i] < -1.5 and fisher[i] >= -1.5:
            fisher_long_signal = True
        
        # Short: Fisher crosses below +1.5 from above
        if fisher_prev[i] > 1.5 and fisher[i] <= 1.5:
            fisher_short_signal = True
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        # Long entry: Fisher long signal + 4h trend bull + macro not strongly bear
        if fisher_long_signal and trend_4h_bull and not macro_bear:
            desired_signal = BASE_SIZE
        
        # Short entry: Fisher short signal + 4h trend bear + macro not strongly bull
        elif fisher_short_signal and trend_4h_bear and not macro_bull:
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
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
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals