#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + volume confirmation + 1w EMA(50) trend filter
# Williams Alligator: Jaw (EMA13,8), Teeth (EMA8,5), Lips (EMA5,3)
# Long when Lips > Teeth > Jaw (bullish alignment) + price > 1w EMA(50) + volume spike
# Short when Lips < Teeth < Jaw (bearish alignment) + price < 1w EMA(50) + volume spike
# Uses 1w EMA(50) for stronger trend alignment to reduce whipsaw in choppy markets
# Designed for low trade frequency (12-37/year on 12h) to minimize fee drag. Works in both bull and bear markets.

name = "12h_WilliamsAlligator_Volume_1wEMA50_v1"
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
    
    # Get 1w data for EMA(50) trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1w for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA to 12h timeframe (wait for completed 1w bar)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator components on 12h
    # Jaw: EMA(13,8) - 13 period EMA smoothed by 8 periods
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean()
    jaw = jaw.ewm(span=8, adjust=False, min_periods=8).mean().values
    
    # Teeth: EMA(8,5) - 8 period EMA smoothed by 5 periods
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean()
    teeth = teeth.ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Lips: EMA(5,3) - 5 period EMA smoothed by 3 periods
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean()
    lips = lips.ewm(span=3, adjust=False, min_periods=3).mean().values
    
    # Volume confirmation (2.0x 20-period average) on 12h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = 60  # max(13+8 for Jaw, 8+5 for Teeth, 5+3 for Lips, 50 for 1w EMA, 20 for volume MA +1 for shift)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Bullish alignment: Lips > Teeth > Jaw
            bullish_alignment = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
            
            # Bearish alignment: Lips < Teeth < Jaw
            bearish_alignment = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
            
            # Long entry: Bullish alignment + price > 1w EMA(50) + volume spike
            if (bullish_alignment and close[i] > ema_50_1w_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Bearish alignment + price < 1w EMA(50) + volume spike
            elif (bearish_alignment and close[i] < ema_50_1w_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bearish alignment OR price below 1w EMA(50)
            bearish_alignment = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
            if bearish_alignment or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bullish alignment OR price above 1w EMA(50)
            bullish_alignment = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
            if bullish_alignment or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals