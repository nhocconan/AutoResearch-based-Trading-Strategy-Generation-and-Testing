#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d EMA34 trend filter + volume spike
# Long when Alligator jaws < teeth < lips (bullish alignment), price > 1d EMA34, and volume > 2x 20-bar average
# Short when Alligator jaws > teeth > lips (bearish alignment), price < 1d EMA34, and volume > 2x 20-bar average
# Williams Alligator identifies trend initiation and alignment; 1d EMA34 filters for higher timeframe trend
# Volume spike confirms momentum behind the move
# Designed for low trade frequency (~12-37/year on 6h) to minimize fee drag
# Works in bull (bullish Alligator alignment with rising volume) and bear (bearish Alligator alignment with rising volume)

name = "6h_WilliamsAlligator_Volume_1dEMA34_Trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator on 6h
    # Jaw (blue line): 13-period SMMA, shifted 8 bars ahead
    jaw = pd.Series(high).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth (red line): 8-period SMMA, shifted 5 bars ahead
    teeth = pd.Series(low).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips (green line): 5-period SMMA, shifted 3 bars ahead
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Volume confirmation (2.0x 20-period average on 6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(34, 13, 8, 5, 20) + 8  # EMA34(1d) + Alligator components + volume MA(20) + jaw shift(8)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Bullish Alligator alignment: jaw < teeth < lips
            bullish_alignment = jaw[i] < teeth[i] and teeth[i] < lips[i]
            # Bearish Alligator alignment: jaw > teeth > lips
            bearish_alignment = jaw[i] > teeth[i] and teeth[i] > lips[i]
            
            # Long entry: bullish alignment, price > 1d EMA34, volume spike
            if (bullish_alignment and 
                close[i] > ema_34_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: bearish alignment, price < 1d EMA34, volume spike
            elif (bearish_alignment and 
                  close[i] < ema_34_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: bearish Alligator alignment or price < 1d EMA34 (trend failure)
            bearish_alignment = jaw[i] > teeth[i] and teeth[i] > lips[i]
            if (bearish_alignment or 
                close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: bullish Alligator alignment or price > 1d EMA34 (trend failure)
            bullish_alignment = jaw[i] < teeth[i] and teeth[i] < lips[i]
            if (bullish_alignment or 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals