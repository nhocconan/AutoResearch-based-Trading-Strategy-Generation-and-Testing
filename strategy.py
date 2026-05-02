#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Uses 6h primary timeframe for lower trade frequency (target: 12-37 trades/year)
# 1d EMA34 ensures alignment with daily trend to avoid counter-trend entries
# Camarilla R3/S3 levels provide clear breakout/breakdown zones from 1d price action
# Volume spike (>2.0 * 20-period EMA) confirms strong institutional participation
# Novelty: 6h timeframe reduces fee drag while Camarilla structure works in both bull/bear markets via trend filter

name = "6h_Camarilla_R3S3_1dEMA34_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for Camarilla pivot and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d Camarilla pivot levels (based on previous 1d bar)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].shift(1).values  # Use previous day for pivot calculation
    
    # Camarilla calculation: based on previous day's range
    rng = high_1d_prev - low_1d_prev
    camarilla_h5 = close_1d_prev + 1.1 * rng / 2  # R4
    camarilla_h4 = close_1d_prev + 1.1 * rng / 4  # R3
    camarilla_h3 = close_1d_prev + 1.1 * rng / 6  # R2
    camarilla_l3 = close_1d_prev - 1.1 * rng / 6  # S2
    camarilla_l2 = close_1d_prev - 1.1 * rng / 4  # S3
    camarilla_l1 = close_1d_prev - 1.1 * rng / 2  # S4
    
    # Align 1d indicators to 6h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    
    # Volume confirmation: volume > 2.0 * 20-period EMA (6h * 4 = ~24 periods)
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need sufficient data for all indicators
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l2_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1d EMA34
        bullish_bias = close[i] > ema_34_1d_aligned[i]
        bearish_bias = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if bullish_bias:
                # Long: price breaks above Camarilla H4 (R3) with volume spike
                if close[i] > camarilla_h4_aligned[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            elif bearish_bias:
                # Short: price breaks below Camarilla L2 (S3) with volume spike
                if close[i] < camarilla_l2_aligned[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid chop around 1d EMA34
        
        elif position == 1:  # Long position
            # Exit: price breaks below Camarilla L3 (S2) or price below 1d EMA34
            if close[i] < camarilla_l2_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Camarilla H4 (R3) or price above 1d EMA34
            if close[i] > camarilla_h4_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals