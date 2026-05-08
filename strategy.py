#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Camarilla_Pivot_Breakout_1wTrend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA20 for trend
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate daily Camarilla pivot levels
    # Camarilla levels for the day: based on previous day's H, L, C
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla levels
    range_val = prev_high - prev_low
    camarilla_r4 = prev_close + range_val * 1.5
    camarilla_s4 = prev_close - range_val * 1.5
    camarilla_r3 = prev_close + range_val * 1.25
    camarilla_s3 = prev_close - range_val * 1.25
    camarilla_r2 = prev_close + range_val * 1.166
    camarilla_s2 = prev_close - range_val * 1.166
    camarilla_r1 = prev_close + range_val * 1.083
    camarilla_s1 = prev_close - range_val * 1.083
    
    # Volume confirmation - 20-day average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ratio[i]) or 
            np.isnan(camarilla_r4[i]) or np.isnan(camarilla_s4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below weekly EMA20
        uptrend = close[i] > ema_20_1w_aligned[i]
        downtrend = close[i] < ema_20_1w_aligned[i]
        
        if position == 0:
            # Long: break above R4 in uptrend + volume
            if (uptrend and 
                close[i] > camarilla_r4[i] and
                vol_ratio[i] > 1.8):
                signals[i] = 0.25
                position = 1
            # Short: break below S4 in downtrend + volume
            elif (downtrend and 
                  close[i] < camarilla_s4[i] and
                  vol_ratio[i] > 1.8):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: break below R2 or trend change
            if (close[i] < camarilla_r2[i] or not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: break above S2 or trend change
            if (close[i] > camarilla_s2[i] or not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals