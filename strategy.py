# State your hypothesis: 1d Weekly Pivot R3/S3 Breakout with 1w Trend Filter and Volume Spike
# Long when price breaks above weekly R3, weekly trend is up (price > weekly EMA34), and volume spikes
# Short when price breaks below weekly S3, weekly trend is down (price < weekly EMA34), and volume spikes
# Exits when price returns to weekly pivot or trend reverses
# Designed for 1d timeframe with weekly structure to capture strong momentum moves while limiting trades

name = "1d_WeeklyPivot_R3S3_Breakout_TrendVolume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter and pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly EMA 34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate weekly pivot points using previous week's OHLC
    w_high = df_1w['high'].values
    w_low = df_1w['low'].values
    w_close = df_1w['close'].values
    
    # Shift to use previous week's data (avoid look-ahead)
    w_high_prev = np.roll(w_high, 1)
    w_low_prev = np.roll(w_low, 1)
    w_close_prev = np.roll(w_close, 1)
    # First period: use current values to avoid NaN
    w_high_prev[0] = w_high[0]
    w_low_prev[0] = w_low[0]
    w_close_prev[0] = w_close[0]
    
    # Calculate weekly pivot point
    w_pivot = (w_high_prev + w_low_prev + w_close_prev) / 3.0
    # Calculate weekly R3 and S3 levels (more extreme levels)
    w_r3 = w_close_prev + (1.1/4) * (w_high_prev - w_low_prev)
    w_s3 = w_close_prev - (1.1/4) * (w_high_prev - w_low_prev)
    
    # Align weekly R3/S3/pivot to 1d timeframe
    w_r3_aligned = align_htf_to_ltf(prices, df_1w, w_r3)
    w_s3_aligned = align_htf_to_ltf(prices, df_1w, w_s3)
    w_pivot_aligned = align_htf_to_ltf(prices, df_1w, w_pivot)
    
    # Volume confirmation (20-period average on 1d)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(w_r3_aligned[i]) or np.isnan(w_s3_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume threshold
        volume_spike = vol_ratio[i] > 2.0
        
        if position == 0:
            # Long: break above weekly R3 + above weekly EMA34 + volume spike
            if (close[i] > w_r3_aligned[i] and 
                close[i] > ema_34_1w_aligned[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: break below weekly S3 + below weekly EMA34 + volume spike
            elif (close[i] < w_s3_aligned[i] and 
                  close[i] < ema_34_1w_aligned[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: return to weekly pivot or trend reversal
            if position == 1:
                # Exit long: price returns to weekly pivot OR trend turns down
                if (close[i] <= w_pivot_aligned[i]) or \
                   (close[i] < ema_34_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to weekly pivot OR trend turns up
                if (close[i] >= w_pivot_aligned[i]) or \
                   (close[i] > ema_34_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals