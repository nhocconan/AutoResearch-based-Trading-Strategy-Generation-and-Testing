#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d RSI(14) trend filter and volume spike
# Uses 1d RSI to filter for trending conditions (RSI > 50 for long, RSI < 50 for short)
# Volume spike (>2.0x 20-bar average) confirms breakout momentum
# ATR-based trailing stop via signal=0 when price retraces 30% of ATR from extreme
# Discrete sizing 0.30 to balance profit potential and fee drag; target 60-120 total trades over 4 years (15-30/year)
# Works in both bull/bear: breakouts capture momentum, RSI filter avoids chop, volume filter ensures participation

name = "4h_Donchian20_1dRSI14_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d RSI(14)
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = np.concatenate([[np.nan] * 13, rsi_1d[13:]])
    
    # Calculate ATR(14) for 4h timeframe (for stoploss)
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume spike filter (>2.0x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma_20)
    
    # Calculate 1d Donchian channels (20-period)
    upper_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align HTF indicators to 4h timeframe (primary)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    upper_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0
    short_extreme = 0.0
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(upper_1d_aligned[i]) or 
            np.isnan(lower_1d_aligned[i]) or np.isnan(atr_4h[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
                short_extreme = 0.0
            continue
        
        if position == 0:
            # Long breakout: price > Upper Donchian AND bullish trend (RSI > 50) AND volume spike
            if close[i] > upper_1d_aligned[i] and rsi_1d_aligned[i] > 50 and volume_filter[i]:
                signals[i] = 0.30
                position = 1
                long_extreme = close[i]
            # Short breakdown: price < Lower Donchian AND bearish trend (RSI < 50) AND volume spike
            elif close[i] < lower_1d_aligned[i] and rsi_1d_aligned[i] < 50 and volume_filter[i]:
                signals[i] = -0.30
                position = -1
                short_extreme = close[i]
        elif position == 1:
            # Update long extreme
            long_extreme = max(long_extreme, close[i])
            # Exit long: price retraces 30% of ATR from extreme
            if close[i] <= long_extreme - 0.30 * atr_4h[i]:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Update short extreme
            short_extreme = min(short_extreme, close[i])
            # Exit short: price retraces 30% of ATR from extreme
            if close[i] >= short_extreme + 0.30 * atr_4h[i]:
                signals[i] = 0.0
                position = 0
                short_extreme = 0.0
            else:
                signals[i] = -0.30
    
    return signals