#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Donchian breakout with weekly trend filter and volume confirmation
# Works in bull via breakouts, in bear via trend-filtered pullbacks to mean
# Target: 15-25 trades/year, low frequency to minimize fee drag
name = "1d_donchian20_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly 50-period EMA for trend
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily 20-period ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Calculate daily Donchian channels (20-period high/low)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 20-period average volume
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_filter = volume[i] > vol_ma[i] if not np.isnan(vol_ma[i]) else True
        
        # Trend filter: price above/below weekly EMA50
        above_weekly_ema = close[i] > ema_50_1w_aligned[i]
        below_weekly_ema = close[i] < ema_50_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price touches opposite band OR trend fails
            if close[i] <= lowest_low[i] or not above_weekly_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price touches opposite band OR trend fails
            if close[i] >= highest_high[i] or not below_weekly_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price breaks above upper band + volume filter + above weekly EMA
            if close[i] > highest_high[i] and vol_filter and above_weekly_ema:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below lower band + volume filter + below weekly EMA
            elif close[i] < lowest_low[i] and vol_filter and below_weekly_ema:
                position = -1
                signals[i] = -0.25
    
    return signals