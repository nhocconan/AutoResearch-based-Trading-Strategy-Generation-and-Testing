#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w EMA50 trend filter and volume confirmation
# Williams Alligator identifies trend via Jaw(13), Teeth(8), Lips(5) SMAs
# When Lips > Teeth > Jaw = bullish alignment, Lips < Teeth < Jaw = bearish alignment
# 1w EMA50 provides higher timeframe trend filter to avoid counter-trend trades
# Volume confirmation (>1.4x average) reduces false signals
# Works in bull/bear: Alligator catches trends in all regimes, volume confirms legitimacy
# Target: 30-80 total trades over 4 years (7-20/year) to minimize fee drag

name = "1d_WilliamsAlligator_1wEMA50_Trend_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator: Jaw(13), Teeth(8), Lips(5) SMAs of median price
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    
    # Alligator alignment conditions
    bullish_alignment = (lips > teeth) & (teeth > jaw)
    bearish_alignment = (lips < teeth) & (teeth < jaw)
    
    # Volume confirmation: volume > 1.4x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_confirm = volume > (1.4 * vol_ma_50)
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 50)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_50[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_bullish = bullish_alignment[i]
        curr_bearish = bearish_alignment[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on Alligator alignment with volume confirmation and trend filter
            if curr_volume_confirm:
                # Bullish: Lips > Teeth > Jaw + price above 1w EMA50
                if curr_bullish and curr_close > curr_ema_50_1w:
                    signals[i] = 0.25
                    position = 1
                # Bearish: Lips < Teeth < Jaw + price below 1w EMA50
                elif curr_bearish and curr_close < curr_ema_50_1w:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: bearish Alligator alignment (Lips < Teeth < Jaw)
            if curr_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: bullish Alligator alignment (Lips > Teeth > Jaw)
            if curr_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals