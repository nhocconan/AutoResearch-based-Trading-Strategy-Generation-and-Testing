#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume confirmation
# Uses 12h primary timeframe for lower trade frequency (target: 12-37/year)
# 1w EMA50 ensures alignment with weekly trend to avoid counter-trend entries
# Camarilla R3/S3 levels from 1d provide precise intraday support/resistance
# Volume spike (>2.0 * 20-period EMA) confirms strong institutional participation
# Novelty: Weekly trend filter + Camarilla structure works in both bull/bear markets by
#          only taking breakouts in direction of higher timeframe trend

name = "12h_Camarilla_R3S3_1wEMA50_Trend_Volume_v1"
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
    
    # 1d HTF data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate 1w HTF data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 12h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla levels from previous completed 1d bar
    # Typical price = (high + low + close) / 3
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    typical_price_1d_series = pd.Series(typical_price_1d.values)
    
    # Pivot point = typical price of previous day
    pivot_1d = typical_price_1d_series.shift(1).values
    
    # Calculate ranges
    hl_1d = df_1d['high'] - df_1d['low']
    hl_1d_series = pd.Series(hl_1d.values)
    
    # Camarilla levels
    R3 = pivot_1d + (hl_1d_series * 1.1 / 4).values
    S3 = pivot_1d - (hl_1d_series * 1.1 / 4).values
    
    # Align Camarilla levels to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume confirmation: volume > 2.0 * 20-period EMA (12h * 2 = ~24 periods)
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need sufficient data for all indicators
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1w EMA50
        bullish_bias = close[i] > ema_50_1w_aligned[i]
        bearish_bias = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if bullish_bias:
                # Long: price breaks above Camarilla R3 with volume spike
                if close[i] > R3_aligned[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            elif bearish_bias:
                # Short: price breaks below Camarilla S3 with volume spike
                if close[i] < S3_aligned[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid chop around 1w EMA50
        
        elif position == 1:  # Long position
            # Exit: price breaks below Camarilla S3 or price below 1w EMA50
            if close[i] < S3_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Camarilla R3 or price above 1w EMA50
            if close[i] > R3_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals