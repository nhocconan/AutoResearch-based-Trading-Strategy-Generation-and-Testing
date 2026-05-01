#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Camarilla levels provide high-probability reversal/breakout points, 1d EMA34 filters for higher timeframe trend,
# volume spike confirms breakout authenticity. Designed for 12h timeframe to target 50-150 total trades over 4 years
# (12-37/year) with low fee drag. Works in both bull and bear markets by following the 1d trend.

name = "12h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # Typical price = (H + L + C) / 3
    typical_price = (high + low + close) / 3.0
    # Range = H - L
    rang = high - low
    
    # Camarilla R3, R2, R1, PP, S1, S2, S3
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # R2 = C + ((H-L) * 1.1/6)
    # R1 = C + ((H-L) * 1.1/12)
    # PP = (H + L + C) / 3
    # S1 = C - ((H-L) * 1.1/12)
    # S2 = C - ((H-L) * 1.1/6)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    
    # We need previous day's H, L, C for today's levels
    # Since we're on 12h timeframe, we calculate daily levels and align
    df_1d_for_camarilla = get_htf_data(prices, '1d')
    if len(df_1d_for_camarilla) < 2:
        return np.zeros(n)
    
    high_1d = df_1d_for_camarilla['high'].values
    low_1d = df_1d_for_camarilla['low'].values
    close_1d_for_camarilla = df_1d_for_camarilla['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    typical_price_1d = (high_1d + low_1d + close_1d_for_camarilla) / 3.0
    rang_1d = high_1d - low_1d
    
    camarilla_pp_1d = typical_price_1d
    camarilla_r1_1d = close_1d_for_camarilla + (rang_1d * 1.1 / 12)
    camarilla_r2_1d = close_1d_for_camarilla + (rang_1d * 1.1 / 6)
    camarilla_r3_1d = close_1d_for_camarilla + (rang_1d * 1.1 / 4)
    camarilla_s1_1d = close_1d_for_camarilla - (rang_1d * 1.1 / 12)
    camarilla_s2_1d = close_1d_for_camarilla - (rang_1d * 1.1 / 6)
    camarilla_s3_1d = close_1d_for_camarilla - (rang_1d * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe (using previous day's levels)
    camarilla_r3_1d_aligned = align_htf_to_ltf(prices, df_1d_for_camarilla, camarilla_r3_1d)
    camarilla_s3_1d_aligned = align_htf_to_ltf(prices, df_1d_for_camarilla, camarilla_s3_1d)
    camarilla_r2_1d_aligned = align_htf_to_ltf(prices, df_1d_for_camarilla, camarilla_r2_1d)
    camarilla_s2_1d_aligned = align_htf_to_ltf(prices, df_1d_for_camarilla, camarilla_s2_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(50, 20)  # Need sufficient history for 1d EMA, Camarilla, and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_1d_aligned[i]) or 
            np.isnan(camarilla_s3_1d_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Camarilla breakout conditions
        breakout_up = close[i] > camarilla_r3_1d_aligned[i]  # Break above R3
        breakout_down = close[i] < camarilla_s3_1d_aligned[i]  # Break below S3
        
        # Trend filter: price above/below 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: upward breakout from R3, volume spike, uptrend
            if breakout_up and vol_spike and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: downward breakout from S3, volume spike, downtrend
            elif breakout_down and vol_spike and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on trend reversal or price re-enters Camarilla (below R2)
            if not uptrend or close[i] < camarilla_r2_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on trend reversal or price re-enters Camarilla (above S2)
            if not downtrend or close[i] > camarilla_s2_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals