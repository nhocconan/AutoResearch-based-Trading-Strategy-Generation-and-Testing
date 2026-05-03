#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla H3L3 breakout with 4h EMA50 trend filter and volume confirmation.
# Long when price breaks above H3 level AND 4h close > 4h EMA50 AND 1h volume > 2.0x 20-period volume MA.
# Short when price breaks below L3 level AND 4h close < 4h EMA50 AND 1h volume > 2.0x 20-period volume MA.
# Exit when price retests the broken level (H3 for longs, L3 for shorts) or trend changes.
# Uses session filter (08-20 UTC) to avoid low-liquidity periods. Position size 0.20.
# Designed for 1h timeframe to achieve 60-150 total trades over 4 years (15-37/year) with strict entry conditions.
# 4h EMA50 filters for higher-timeframe alignment, volume confirms institutional participation.

name = "1h_Camarilla_H3L3_Breakout_4hEMA50_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend direction
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels from previous 1d
    # H3 = C + 1.1*(H-L)/4, L3 = C - 1.1*(H-L)/4
    # where C, H, L are from previous 1d candle
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    df_1d_close = df_1d['close'].values
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    
    # Previous day's values (for today's Camarilla levels)
    prev_close = np.roll(df_1d_close, 1)
    prev_high = np.roll(df_1d_high, 1)
    prev_low = np.roll(df_1d_low, 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Camarilla H3 and L3 levels
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 4
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 4
    
    # Align Camarilla levels to 1h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate 1h volume 20-period MA for spike detection
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(volume_ma_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        h3_level = camarilla_h3_aligned[i]
        l3_level = camarilla_l3_aligned[i]
        
        # Volume spike condition: current 1h volume > 2.0x 20-period volume MA
        volume_spike = volume[i] > (volume_ma_20[i] * 2.0)
        
        # 4h trend conditions
        trend_up = close_val > ema_50_4h_aligned[i]   # 4h uptrend
        trend_down = close_val < ema_50_4h_aligned[i]  # 4h downtrend
        
        if position == 0:
            # Long: Price breaks above H3 AND 4h uptrend AND volume spike AND session
            if close_val > h3_level and trend_up and volume_spike:
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below L3 AND 4h downtrend AND volume spike AND session
            elif close_val < l3_level and trend_down and volume_spike:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: Price retests H3 level OR trend changes to down
            if close_val < h3_level or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: Price retests L3 level OR trend changes to up
            if close_val > l3_level or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals