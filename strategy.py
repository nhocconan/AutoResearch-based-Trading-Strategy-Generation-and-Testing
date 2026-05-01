#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike (>2.0x 20-bar volume MA)
# Camarilla R3/S3 levels act as strong intraday support/resistance; breakouts with volume indicate institutional interest.
# 1d EMA34 ensures alignment with higher timeframe trend to avoid counter-trend trades.
# Volume threshold increased to 2.0x to reduce false breakouts and lower trade frequency.
# Discrete sizing (0.25) minimizes fee churn. Target: 75-200 total trades over 4 years (19-50/year).
# Works in bull (breakouts with volume) and bear (failed reversals reverse quickly at strong levels).

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA(34) on 1d close
    ema_1d_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 4h timeframe
    ema_1d_34_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_34)
    
    # Calculate Camarilla levels for each 1d bar using prior 1d OHLC (standard Camarilla)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    rango = high_1d - low_1d
    camarilla_r3 = close_1d + 1.1 * rango * 1.1 / 4  # R3 = C + 1.1*(H-L)*1.1/4
    camarilla_s3 = close_1d - 1.1 * rango * 1.1 / 4  # S3 = C - 1.1*(H-L)*1.1/4
    camarilla_r4 = close_1d + 1.1 * rango * 1.1 / 2  # R4 = C + 1.1*(H-L)*1.1/2
    camarilla_s4 = close_1d - 1.1 * rango * 1.1 / 2  # S4 = C - 1.1*(H-L)*1.1/2
    
    # Align Camarilla levels to 4h timeframe (use prior completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 34  # Need 34 for 1d EMA and 20 for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1d_34_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Camarilla breakout conditions (using prior bar levels to avoid look-ahead)
        breakout_up = curr_close > camarilla_r3_aligned[i-1]  # Break above R3
        breakout_down = curr_close < camarilla_s3_aligned[i-1]  # Break below S3
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Camarilla breakout up, price above 1d EMA34, volume spike
            if breakout_up and curr_close > ema_1d_34_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Camarilla breakout down, price below 1d EMA34, volume spike
            elif breakout_down and curr_close < ema_1d_34_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Camarilla breakdown or price below 1d EMA34
            if curr_close < camarilla_s3_aligned[i] or curr_close < ema_1d_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Camarilla breakout or price above 1d EMA34
            if curr_close > camarilla_r3_aligned[i] or curr_close > ema_1d_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals