#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume spike confirmation
# Camarilla pivots provide institutional support/resistance levels from prior 1d session
# Breakouts above R3 or below S3 with volume spike indicate institutional participation
# 1w EMA50 filter ensures trading with higher timeframe trend
# Works in bull/bear by only taking breakouts in direction of 1w trend
# Target: 12-37 trades/year (50-150 over 4 years) to minimize fee drag

name = "12h_Camarilla_R3_S3_Breakout_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1d data for Camarilla pivot calculation (prior day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior 1d bar
    # R4 = close + ((high-low)*1.1/2)
    # R3 = close + ((high-low)*1.1/4)
    # S3 = close - ((high-low)*1.1/4)
    # S4 = close - ((high-low)*1.1/2)
    camarilla_R3 = df_1d['close'] + ((df_1d['high'] - df_1d['low']) * 1.1 / 4)
    camarilla_S3 = df_1d['close'] - ((df_1d['high'] - df_1d['low']) * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe (use prior day's levels)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3.values)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3.values)
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(50, 20)  # Need sufficient history for 1w EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(camarilla_R3_aligned[i]) or 
            np.isnan(camarilla_S3_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1w EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above R3, volume spike, uptrend
            if close[i] > camarilla_R3_aligned[i] and vol_spike and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3, volume spike, downtrend
            elif close[i] < camarilla_S3_aligned[i] and vol_spike and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on trend reversal or price re-enters R3-S3 range
            if not uptrend or (camarilla_S3_aligned[i] <= close[i] <= camarilla_R3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on trend reversal or price re-enters R3-S3 range
            if not downtrend or (camarilla_S3_aligned[i] <= close[i] <= camarilla_R3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals