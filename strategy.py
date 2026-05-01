#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Uses Camarilla pivot levels from daily data for precise entry/exit points
# EMA34 on 1d determines structural trend bias (long above, short below)
# Volume confirmation > 2.0x 20-period EMA ensures institutional participation
# Designed for low trade frequency: ~20-40 trades/year per symbol with 0.30 sizing
# Camarilla levels work in both bull and bear markets as dynamic support/resistance
# Breakouts in direction of 1d trend have higher follow-through probability

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for Camarilla calculation and EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate daily OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from prior day OHLC
    # Camarilla R3 = Close + (High - Low) * 1.1/4
    # Camarilla S3 = Close - (High - Low) * 1.1/4
    # Camarilla R4 = Close + (High - Low) * 1.1/2
    # Camarilla S4 = Close - (High - Low) * 1.1/2
    cam_r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    cam_s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    cam_r4 = close_1d + (high_1d - low_1d) * 1.1 / 2
    cam_s4 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe (use prior day's levels)
    cam_r3_aligned = align_htf_to_ltf(prices, df_1d, cam_r3)
    cam_s3_aligned = align_htf_to_ltf(prices, df_1d, cam_s3)
    cam_r4_aligned = align_htf_to_ltf(prices, df_1d, cam_r4)
    cam_s4_aligned = align_htf_to_ltf(prices, df_1d, cam_s4)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 2.0 * 20-period EMA
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need 1d EMA34 (34 bars) + Camarilla (2 days) + volume EMA20
    start_idx = max(34, 2, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(cam_r3_aligned[i]) or np.isnan(cam_s3_aligned[i]) or 
            np.isnan(cam_r4_aligned[i]) or np.isnan(cam_s4_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1d EMA34
        bullish_bias = close[i] > ema_34_aligned[i]
        bearish_bias = close[i] < ema_34_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if bullish_bias:
                # Long: price breaks above Camarilla R3 with volume spike
                if close[i] > cam_r3_aligned[i] and volume_spike[i]:
                    signals[i] = 0.30
                    position = 1
                else:
                    signals[i] = 0.0
            elif bearish_bias:
                # Short: price breaks below Camarilla S3 with volume spike
                if close[i] < cam_s3_aligned[i] and volume_spike[i]:
                    signals[i] = -0.30
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid chop around EMA
        
        elif position == 1:  # Long position
            # Exit: price reaches Camarilla R4 (take profit) or breaks below S3 (stop)
            if close[i] >= cam_r4_aligned[i] or close[i] < cam_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: price reaches Camarilla S4 (take profit) or breaks above R3 (stop)
            if close[i] <= cam_s4_aligned[i] or close[i] > cam_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals