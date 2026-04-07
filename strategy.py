#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: Daily Donchian breakout with weekly trend filter and volume confirmation
# Works in bull markets via breakouts above upper band, in bear via mean reversion at lower band
# Weekly trend filter avoids counter-trend trades in strong trends
# Volume confirmation ensures institutional participation
# Target: 15-25 trades/year, very low frequency to minimize fee drag
name = "1d_donchian20_1w_trend_volume_v4"
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
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly 50-period EMA for trend filter
    close_1w = df_1w['close'].values
    ema_50w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50w_aligned = align_htf_to_ltf(prices, df_1w, ema_50w)
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily 20-period volume moving average
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate ATR(14) for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], 
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period high/low)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(ema_50w_aligned[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current daily volume > 20-day average volume
        vol_confirm = volume[i] > vol_ma_1d_aligned[i]
        
        # Volatility filter: only trade when ATR is above its 50-period average
        atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
        vol_filter = atr[i] > atr_ma[i] if not np.isnan(atr_ma[i]) else True
        
        # Trend filter: weekly EMA50 slope
        if i >= 21:
            ema_slope = ema_50w_aligned[i] - ema_50w_aligned[i-1]
        else:
            ema_slope = 0
        
        if position == 1:  # Long position
            # Exit: price touches lower band OR volatility drops OR trend turns down
            if close[i] <= lowest_low[i] or not vol_filter or ema_slope < 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price touches upper band OR volatility drops OR trend turns up
            if close[i] >= highest_high[i] or not vol_filter or ema_slope > 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Enter long: price breaks above upper band + volume confirmation + volatility filter + uptrend
            if (close[i] > highest_high[i] and vol_confirm and vol_filter and 
                ema_slope > 0):
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below lower band + volume confirmation + volatility filter + downtrend
            elif (close[i] < lowest_low[i] and vol_confirm and vol_filter and 
                  ema_slope < 0):
                position = -1
                signals[i] = -0.25
    
    return signals