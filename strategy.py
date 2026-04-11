#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with weekly trend filter and volume confirmation
# Long when price breaks above Camarilla R3 level + volume > 2x average + weekly trend up
# Short when price breaks below Camarilla S3 level + volume > 2x average + weekly trend down
# Exit when price returns to Camarilla pivot point or weekly trend reverses
# Designed for 30-100 trades over 4 years on 1d timeframe with strong trend capture and low turnover

name = "1d_1w_camarilla_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate 30-day average volume for volume filter
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    # Calculate Camarilla pivot levels from previous day
    # Using previous day's OHLC to calculate today's levels
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    # Set first values to avoid NaN
    prev_close[0] = close[0]
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Camarilla levels
    r3 = pivot + (range_val * 1.1 / 4)
    s3 = pivot - (range_val * 1.1 / 4)
    r4 = pivot + (range_val * 1.1 / 2)
    s4 = pivot - (range_val * 1.1 / 2)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(vol_ma_30[i]) or np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 2x 30-day average
        volume_filter = volume[i] > 2.0 * vol_ma_30[i]
        
        # Trend filter: price relative to weekly EMA20
        is_uptrend = close[i] > ema_20_1w_aligned[i]
        is_downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Entry conditions
        long_breakout = close[i] > r3[i] and volume_filter and is_uptrend
        short_breakout = close[i] < s3[i] and volume_filter and is_downtrend
        
        # Exit conditions - return to pivot or trend reversal
        long_exit = (close[i] < pivot[i]) or (not is_uptrend)
        short_exit = (close[i] > pivot[i]) or (not is_downtrend)
        
        # Priority: entry > exit > hold
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals