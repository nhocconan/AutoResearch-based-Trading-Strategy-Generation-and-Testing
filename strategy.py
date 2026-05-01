#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Camarilla pivot levels provide institutional support/resistance, 1d EMA34 filters higher timeframe trend,
# volume spike confirms breakout strength. Designed for low trade frequency (~50-150 total 6h trades)
# to minimize fee drag and work in both bull and bear markets via trend following.

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d OHLC for Camarilla levels (using previous completed 1d bar)
    # We need to get the 1d OHLC values and align them properly
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    # Camarilla R3 = close + 1.1*(high-low)/2
    # Camarilla S3 = close - 1.1*(high-low)/2
    # Camarilla R4 = close + 1.1*(high-low)
    # Camarilla S4 = close - 1.1*(high-low)
    rng = high_1d - low_1d
    camarilla_r3 = close_1d_vals + 1.1 * rng / 2.0
    camarilla_s3 = close_1d_vals - 1.1 * rng / 2.0
    camarilla_r4 = close_1d_vals + 1.1 * rng
    camarilla_s4 = close_1d_vals - 1.1 * rng
    
    # Align Camarilla levels to 6h timeframe (wait for 1d bar to close)
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
    start_idx = max(34, 20)  # Need sufficient history for 1d EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions using Camarilla levels
        breakout_up = close[i] > camarilla_r3_aligned[i-1]  # Break above R3
        breakout_down = close[i] < camarilla_s3_aligned[i-1]  # Break below S3
        
        # Trend filter: price above/below 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: upward breakout above R3, volume spike, uptrend
            if breakout_up and vol_spike and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: downward breakout below S3, volume spike, downtrend
            elif breakout_down and vol_spike and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on trend reversal or price re-enters Camarilla S3-R3 range
            if not uptrend or close[i] < camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on trend reversal or price re-enters Camarilla S3-R3 range
            if not downtrend or close[i] > camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals