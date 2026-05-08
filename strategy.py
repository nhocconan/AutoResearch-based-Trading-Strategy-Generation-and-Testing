#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Weekly_Camarilla_R3S3_Breakout_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once for trend filter and volatility context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly EMA20 trend filter
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_1w = (close_1w > ema20_1w).astype(float)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # Get daily data once for ATR calculation (used in position sizing and volatility filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Get previous day's OHLC for Camarilla pivot levels (R3/S3)
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close = np.roll(df_1d['close'].values, 1)
    # Handle first day: use current day's values as previous
    prev_high[0] = df_1d['high'].values[0]
    prev_low[0] = df_1d['low'].values[0]
    prev_close[0] = df_1d['close'].values[0]
    
    # Camarilla pivot levels calculation (R3 and S3)
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    r3 = pivot + (range_val * 1.1 / 2)  # R3 level
    s3 = pivot - (range_val * 1.1 / 2)  # S3 level
    
    # Align Camarilla levels to daily timeframe (no alignment needed as already daily)
    r3_1d = r3
    s3_1d = s3
    
    # Volume spike detection: current volume > 2.0 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma20 * 2.0)
    
    # Volatility filter: require ATR > 0.5 * 50-period average ATR (avoid low volatility chop)
    atr_ma50 = pd.Series(atr14).rolling(window=50, min_periods=50).mean().values
    vol_filter = atr14 > (atr_ma50 * 0.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_1d[i]) or np.isnan(s3_1d[i]) or np.isnan(trend_1w_aligned[i]) or 
            np.isnan(vol_ma20[i]) or np.isnan(atr14[i]) or np.isnan(atr_ma50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above R3 with volume spike, weekly uptrend, and sufficient volatility
            long_cond = (close[i] > r3_1d[i] and vol_spike[i] and trend_1w_aligned[i] > 0.5 and vol_filter[i])
            
            # Short entry: price breaks below S3 with volume spike, weekly downtrend, and sufficient volatility
            short_cond = (close[i] < s3_1d[i] and vol_spike[i] and trend_1w_aligned[i] < 0.5 and vol_filter[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverses back below S3 (mean reversion to opposite level)
            if close[i] < s3_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverses back above R3 (mean reversion to opposite level)
            if close[i] > r3_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly Camarilla R3/S3 breakout with volume spike, volatility filter, and weekly trend alignment.
# Uses 1d timeframe with weekly EMA20 trend filter for multi-timeframe alignment.
# Entry conditions: price breaks weekly Camarilla R3/S3 levels + volume spike (2x 20-day MA) + weekly trend alignment + volatility filter (ATR > 0.5x 50-day MA ATR).
# Exit conditions: mean reversion to opposite Camarilla level (long exit when price < S3, short exit when price > R3).
# Position size: 0.25 to manage risk and reduce drawdown.
# Designed for low trade frequency (target: 10-25 trades/year) to minimize fee drag while capturing significant breakouts in both bull and bear markets.