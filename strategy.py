#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + 1w EMA Trend + Volume Spike
# Williams Alligator identifies trend phases: Jaw (TEETH=13), Teeth (TEETH=8), Lips (TEETH=5) SMAs.
# When Lips cross above Teeth and Jaw = bullish alignment (all three rising).
# When Lips cross below Teeth and Jaw = bearish alignment (all three falling).
# Combined with 1w EMA50 trend filter and volume confirmation (>2.0x 20-bar average).
# Exit on Alligator cross reversal or when price closes outside Alligator mouth.
# Uses discrete position sizing (0.25) to limit drawdown and reduce fee churn.
# Target: 30-100 total trades over 4 years (7-25/year) on 1d timeframe.
# Works in both bull/bear markets by requiring alignment with 1w trend.
# Volume confirmation filters weak signals.

name = "1d_WilliamsAlligator_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator SMAs on 1d data
    close_1d = df_1d['close'].values
    # Lips = SMA(5), Teeth = SMA(8), Jaw = SMA(13)
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    
    # Align Alligator lines to 1d timeframe (no additional delay needed for SMAs)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13)  # Ensure sufficient history for volume MA and Alligator
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(lips_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(jaw_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 1w EMA trend filter
        ema_trend_up = close[i] > ema_50_1w_aligned[i]
        ema_trend_down = close[i] < ema_50_1w_aligned[i]
        
        price = close[i]
        
        # Williams Alligator conditions
        # Bullish alignment: Lips > Teeth > Jaw and all rising
        lips_rising = lips_aligned[i] > lips_aligned[i-1]
        teeth_rising = teeth_aligned[i] > teeth_aligned[i-1]
        jaw_rising = jaw_aligned[i] > jaw_aligned[i-1]
        bullish_align = (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]) and lips_rising and teeth_rising and jaw_rising
        
        # Bearish alignment: Lips < Teeth < Jaw and all falling
        lips_falling = lips_aligned[i] < lips_aligned[i-1]
        teeth_falling = teeth_aligned[i] < teeth_aligned[i-1]
        jaw_falling = jaw_aligned[i] < jaw_aligned[i-1]
        bearish_align = (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]) and lips_falling and teeth_falling and jaw_falling
        
        # Mouth conditions: price outside Alligator mouth
        above_mouth = price > lips_aligned[i] and price > teeth_aligned[i] and price > jaw_aligned[i]
        below_mouth = price < lips_aligned[i] and price < teeth_aligned[i] and price < jaw_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: Bullish alignment, 1w EMA50 uptrend, volume confirm, price above mouth
            if bullish_align and ema_trend_up and vol_confirm and above_mouth:
                signals[i] = 0.25
                position = 1
            # Short entry: Bearish alignment, 1w EMA50 downtrend, volume confirm, price below mouth
            elif bearish_align and ema_trend_down and vol_confirm and below_mouth:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on bearish alignment or price below mouth
            if bearish_align or below_mouth:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on bullish alignment or price above mouth
            if bullish_align or above_mouth:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals