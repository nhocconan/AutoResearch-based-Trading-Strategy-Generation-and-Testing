#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + volume confirmation + 1d EMA(34) trend filter
# Williams Alligator: Jaw (EMA13,8), Teeth (EMA8,5), Lips (EMA5,3)
# Long when Lips > Teeth > Jaw (bullish alignment) + price > 1d EMA(34) + volume spike
# Short when Lips < Teeth < Jaw (bearish alignment) + price < 1d EMA(34) + volume spike
# Uses 1d EMA(34) for stronger trend alignment to reduce whipsaw in choppy markets
# Designed for low trade frequency (12-37/year on 6h) to minimize fee drag. Works in both bull and bear markets.

name = "6h_WilliamsAlligator_Volume_1dEMA34_v1"
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
    
    # Get 1d data for EMA(34) trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 6h timeframe (wait for completed 1d bar)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator components on 6h
    # Jaw: EMA(13,8) - 13 period EMA smoothed by 8 periods
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean()
    jaw = jaw.ewm(span=8, adjust=False, min_periods=8).mean().values
    
    # Teeth: EMA(8,5) - 8 period EMA smoothed by 5 periods
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean()
    teeth = teeth.ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Lips: EMA(5,3) - 5 period EMA smoothed by 3 periods
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean()
    lips = lips.ewm(span=3, adjust=False, min_periods=3).mean().values
    
    # Volume confirmation (2.0x 20-period average) on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = 50  # max(13+8 for Jaw, 8+5 for Teeth, 5+3 for Lips, 34 for 1d EMA, 20 for volume MA +1 for shift)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Bullish alignment: Lips > Teeth > Jaw
            bullish_alignment = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
            
            # Bearish alignment: Lips < Teeth < Jaw
            bearish_alignment = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
            
            # Long entry: Bullish alignment + price > 1d EMA(34) + volume spike
            if (bullish_alignment and close[i] > ema_34_1d_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Bearish alignment + price < 1d EMA(34) + volume spike
            elif (bearish_alignment and close[i] < ema_34_1d_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bearish alignment OR price below 1d EMA(34)
            bearish_alignment = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
            if bearish_alignment or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bullish alignment OR price above 1d EMA(34)
            bullish_alignment = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
            if bullish_alignment or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals