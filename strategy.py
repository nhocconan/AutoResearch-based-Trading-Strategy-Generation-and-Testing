#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA34 trend filter and volume spike (>1.8x 24-bar MA)
# Uses Camarilla pivot levels from 1d to identify key support/resistance zones.
# Breakouts above R1 or below S1 with volume confirmation capture momentum.
# 4h EMA34 ensures alignment with short-medium term trend to reduce whipsaws.
# Volume spike (>1.8x) confirms participation. Discrete sizing (0.20) minimizes fee churn.
# Session filter (08-20 UTC) reduces noise trades. Target: 60-150 total trades over 4 years (15-37/year).

name = "1h_Camarilla_R1S1_Breakout_4hEMA34_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours for UTC 08-20 filter
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h HTF data for EMA calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # 4h EMA(34) on 4h close
    ema_4h_34 = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 4h EMA to 1h timeframe
    ema_4h_34_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_34)
    
    # 1d data for Camarilla pivot levels (using previous 1d bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    # R1 = C + ((H-L)*1.1/12), S1 = C - ((H-L)*1.1/12)
    prev_1d_close = df_1d['close'].shift(1).values
    prev_1d_high = df_1d['high'].shift(1).values
    prev_1d_low = df_1d['low'].shift(1).values
    
    camarilla_range = (prev_1d_high - prev_1d_low) * 1.1
    camarilla_r1 = prev_1d_close + camarilla_range / 12
    camarilla_s1 = prev_1d_close - camarilla_range / 12
    
    # Align Camarilla levels to 1h timeframe (they change only when 1d bar changes)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: current volume > 1.8 * 24-period average volume
    volume_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (volume_ma_24 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 24  # Need 24 for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_4h_34_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(volume_ma_24[i])):
            signals[i] = 0.0
            continue
        
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla R1, above 4h EMA, and volume spike
            if curr_high > camarilla_r1_aligned[i] and curr_close > ema_4h_34_aligned[i] and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Camarilla S1, below 4h EMA, and volume spike
            elif curr_low < camarilla_s1_aligned[i] and curr_close < ema_4h_34_aligned[i] and vol_spike:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on price breaking below Camarilla S1 or below 4h EMA
            if curr_low < camarilla_s1_aligned[i] or curr_close < ema_4h_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit on price breaking above Camarilla R1 or above 4h EMA
            if curr_high > camarilla_r1_aligned[i] or curr_close > ema_4h_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals