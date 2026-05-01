#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator (Jaw/Teeth/Lips) + 1d EMA50 trend filter + volume confirmation.
# The Alligator identifies trending vs ranging markets via SMAs (Jaw=13, Teeth=8, Lips=5).
# Long when Lips > Teeth > Jaw (bullish alignment) and price above 1d EMA50 with volume spike.
# Short when Lips < Teeth < Jaw (bearish alignment) and price below 1d EMA50 with volume spike.
# Works in both bull and bear markets by only trading in the direction of the higher timeframe trend.
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing (0.25-0.30).

name = "12h_WilliamsAlligator_1dEMA50_Trend_VolumeConfirm_v1"
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
    
    # Daily HTF data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator: SMAs of median price (typical price = (H+L+C)/3)
    typical_price = (high + low + close) / 3.0
    jaw = pd.Series(typical_price).rolling(window=13, min_periods=13).mean().values  # 13-period SMA
    teeth = pd.Series(typical_price).rolling(window=8, min_periods=8).mean().values    # 8-period SMA
    lips = pd.Series(typical_price).rolling(window=5, min_periods=5).mean().values    # 5-period SMA
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 20  # Need 20 for volume MA, 13 for Jaw
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_12h[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(volume_ma_20[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Bullish alignment: Lips > Teeth > Jaw
            bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
            # Bearish alignment: Lips < Teeth < Jaw
            bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
            
            # Long: Bullish alignment, above daily EMA50, and volume confirmation
            if bullish_alignment and curr_close > ema_50_12h[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment, below daily EMA50, and volume confirmation
            elif bearish_alignment and curr_close < ema_50_12h[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on loss of bullish alignment or price below daily EMA50
            if not (lips[i] > teeth[i] and teeth[i] > jaw[i]) or curr_close < ema_50_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on loss of bearish alignment or price above daily EMA50
            if not (lips[i] < teeth[i] and teeth[i] < jaw[i]) or curr_close > ema_50_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals