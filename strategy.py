#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Uses Camarilla pivot levels from 1d timeframe for high-probability breakout entries
# 1d EMA34 provides trend filter to avoid counter-trend trades
# Volume > 2.0x 20-period average confirms institutional participation and reduces false breakouts
# Discrete position sizing (0.25) with Camarilla H3/L3 exit for quick profit taking
# Designed for ~12-37 trades/year on 6h timeframe to minimize fee drag while capturing strong moves
# Works in bull/bear via trend filter - only trades in direction of 1d EMA34

name = "6h_Camarilla_R3S3_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (based on previous day's OHLC)
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # Formula based on previous day's range: R = High - Low
    # H3 = Close + R * 1.1/4, L3 = Close - R * 1.1/4
    # H4 = Close + R * 1.1/2, L4 = Close - R * 1.1/2
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    camarilla_h3 = prev_close + prev_range * 1.1 / 4
    camarilla_l3 = prev_close - prev_range * 1.1 / 4
    camarilla_h4 = prev_close + prev_range * 1.1 / 2
    camarilla_l4 = prev_close - prev_range * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe (completed 1d bar only)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6h Donchian channels (10-period) for exit
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper_10 = high_series.rolling(window=10, min_periods=10).max().values
    donchian_lower_10 = low_series.rolling(window=10, min_periods=10).min().values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # 1d EMA34 and volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(donchian_upper_10[i]) or 
            np.isnan(donchian_lower_10[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_h3 = camarilla_h3_aligned[i]
        curr_l3 = camarilla_l3_aligned[i]
        curr_h4 = camarilla_h4_aligned[i]
        curr_l4 = camarilla_l4_aligned[i]
        curr_ema34_1d = ema_34_1d_aligned[i]
        curr_donchian_upper_10 = donchian_upper_10[i]
        curr_donchian_lower_10 = donchian_lower_10[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price below 10-period Donchian lower (quick profit taking)
            if curr_close < curr_donchian_lower_10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above 10-period Donchian upper (quick profit taking)
            if curr_close > curr_donchian_upper_10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 2.0x 20-period average
            vol_confirmed = curr_volume > 2.0 * curr_vol_ma
            
            # Long when price breaks above Camarilla H3, 1d EMA34 up-trend, volume confirmed
            if curr_high > curr_h3 and curr_close > curr_ema34_1d and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Camarilla L3, 1d EMA34 down-trend, volume confirmed
            elif curr_low < curr_l3 and curr_close < curr_ema34_1d and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals