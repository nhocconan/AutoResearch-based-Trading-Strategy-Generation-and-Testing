#!/usr/bin/env python3
"""
4h_Pivot_R1_S1_Breakout_Volume_Momentum_v1
4h Camarilla pivot (R1/S1) breakout with volume confirmation and momentum filter.
Trades breakouts from daily Camarilla levels with volume surge and RSI momentum.
Designed for 4h timeframe with 12h trend filter to avoid counter-trend trades.
Target: 20-50 trades/year (80-200 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Daily Camarilla Pivot Levels ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R1, S1, R2, S2, R3, S3, R4, S4
    # Formula: Range = (high - low)
    # R4 = close + range * 1.5000
    # R3 = close + range * 1.2500
    # R2 = close + range * 1.1666
    # R1 = close + range * 1.0833
    # S1 = close - range * 1.0833
    # S2 = close - range * 1.1666
    # S3 = close - range * 1.2500
    # S4 = close - range * 1.5000
    camarilla_r1 = np.zeros_like(close_1d)
    camarilla_s1 = np.zeros_like(close_1d)
    camarilla_r2 = np.zeros_like(close_1d)
    camarilla_s2 = np.zeros_like(close_1d)
    camarilla_r3 = np.zeros_like(close_1d)
    camarilla_s3 = np.zeros_like(close_1d)
    camarilla_r4 = np.zeros_like(close_1d)
    camarilla_s4 = np.zeros_like(close_1d)
    
    for i in range(len(close_1d)):
        if i >= 1:  # Need previous day's data
            prev_high = high_1d[i-1]
            prev_low = low_1d[i-1]
            prev_close = close_1d[i-1]
            rng = prev_high - prev_low
            
            camarilla_r1[i] = prev_close + rng * 1.0833
            camarilla_s1[i] = prev_close - rng * 1.0833
            camarilla_r2[i] = prev_close + rng * 1.1666
            camarilla_s2[i] = prev_close - rng * 1.1666
            camarilla_r3[i] = prev_close + rng * 1.2500
            camarilla_s3[i] = prev_close - rng * 1.2500
            camarilla_r4[i] = prev_close + rng * 1.5000
            camarilla_s4[i] = prev_close - rng * 1.5000
        else:
            camarilla_r1[i] = np.nan
            camarilla_s1[i] = np.nan
            camarilla_r2[i] = np.nan
            camarilla_s2[i] = np.nan
            camarilla_r3[i] = np.nan
            camarilla_s3[i] = np.nan
            camarilla_r4[i] = np.nan
            camarilla_s4[i] = np.nan
    
    # === Align Camarilla levels to 4h timeframe ===
    r1_4h = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    r2_4h = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    s2_4h = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    r3_4h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_4h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    r4_4h = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_4h = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # === 12h Trend Filter (EMA34) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_4h = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # === 4h RSI (14-period) for momentum ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    for i in range(len(close)):
        if i >= 14:
            if i == 14:
                avg_gain[i] = np.mean(gain[1:15])
                avg_loss[i] = np.mean(loss[1:15])
            else:
                avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
                avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === 4h Volume Confirmation (20-period average) ===
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 20:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    vol_confirm = volume > vol_ma_20 * 1.5  # volume spike: 1.5x average
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(ema_12h_4h[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above R1 with volume and momentum
            if (close[i] > r1_4h[i] and 
                vol_confirm[i] and 
                rsi[i] > 50 and  # momentum filter
                close[i] > ema_12h_4h[i]):  # trend filter
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below S1 with volume and momentum
            elif (close[i] < s1_4h[i] and 
                  vol_confirm[i] and 
                  rsi[i] < 50 and  # momentum filter
                  close[i] < ema_12h_4h[i]):  # trend filter
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price reaches R2 or RSI overbought
            if (close[i] >= r2_4h[i] or 
                rsi[i] > 70):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reaches S2 or RSI oversold
            if (close[i] <= s2_4h[i] or 
                rsi[i] < 30):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Pivot_R1_S1_Breakout_Volume_Momentum_v1"
timeframe = "4h"
leverage = 1.0