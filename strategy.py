#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + volume confirmation + 1w EMA(34) trend filter
# Williams Alligator: Jaw=EMA(13,8), Teeth=EMA(8,5), Lips=EMA(5,3)
# Long when Lips > Teeth > Jaw (bullish alignment) + price > 1w EMA(34) + volume spike
# Short when Lips < Teeth < Jaw (bearish alignment) + price < 1w EMA(34) + volume spike
# Uses 1w EMA(34) for stronger trend alignment to reduce whipsaw in choppy markets
# Designed for low trade frequency (7-25/year) to minimize fee drag. Works in both bull and bear markets.

name = "1d_WilliamsAlligator_Volume_1wEMA34_v1"
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
    
    # Get 1w data for EMA(34) trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1w for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA to 1d timeframe (wait for completed 1w bar)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Williams Alligator components on 1d
    # Jaw: EMA(13,8) - 13 period EMA with 8 period shift
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().shift(8).values
    # Teeth: EMA(8,5) - 8 period EMA with 5 period shift
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().shift(5).values
    # Lips: EMA(5,3) - 5 period EMA with 3 period shift
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().shift(3).values
    
    # Volume confirmation (2.0x 20-period average) on 1d
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = 40  # max(13+8 for Jaw, 8+5 for Teeth, 5+3 for Lips, 34 for 1w EMA, 20 for volume MA +1 for shift)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Bullish alignment: Lips > Teeth > Jaw
            bullish_alignment = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
            # Bearish alignment: Lips < Teeth < Jaw
            bearish_alignment = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
            
            # Long entry: Bullish alignment + price > 1w EMA(34) + volume spike
            if (bullish_alignment and close[i] > ema_34_1w_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Bearish alignment + price < 1w EMA(34) + volume spike
            elif (bearish_alignment and close[i] < ema_34_1w_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bearish alignment OR price below 1w EMA(34)
            bearish_alignment = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
            if bearish_alignment or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bullish alignment OR price above 1w EMA(34)
            bullish_alignment = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
            if bullish_alignment or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals