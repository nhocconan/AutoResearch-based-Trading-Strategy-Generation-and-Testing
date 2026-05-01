#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator (Jaw/Teeth/Lips) + 1d EMA50 trend filter + volume spike confirmation
# Alligator identifies trendless periods (lines intertwined) vs trending (lines separated, ordered).
# Works in bull/bear: in trends, trade direction of Alligator alignment; in ranges, avoid false signals.
# Target: 20-50 trades/year (80-200 over 4 years) to minimize fee drag

name = "4h_WilliamsAlligator_1dEMA50_Trend_VolumeSpike_v1"
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
    
    # 1d HTF data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator: SMAs of median price (typical price = (H+L+C)/3)
    typical_price = (high + low + close) / 3.0
    # Jaw: 13-period SMA, shifted 8 bars
    jaw = pd.Series(typical_price).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMA, shifted 5 bars
    teeth = pd.Series(typical_price).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMA, shifted 3 bars
    lips = pd.Series(typical_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(13+8, 8+5, 5+3, 20, 50)  # Need sufficient history for Alligator shifts, volume MA, and 1d EMA
    
    for i in range(start_idx, n):
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Alligator conditions: lines separated and ordered
        # Bullish alignment: Lips > Teeth > Jaw (all rising)
        bullish_aligned = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        # Bearish alignment: Lips < Teeth < Jaw (all falling)
        bearish_aligned = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Bullish Alligator alignment, volume spike, uptrend
            if bullish_aligned and vol_spike and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator alignment, volume spike, downtrend
            elif bearish_aligned and vol_spike and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on bearish alignment or trend reversal
            if bearish_aligned or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on bullish alignment or trend reversal
            if bullish_aligned or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals