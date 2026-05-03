#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1w trend filter and volume confirmation
# Long when Alligator is bullish (jaw < teeth < lips), price > 1w EMA34, and volume spike
# Short when Alligator is bearish (jaw > teeth > lips), price < 1w EMA34, and volume spike
# Uses Williams Alligator from 6h for trend/momentum, 1w EMA for higher timeframe trend, volume for confirmation
# Designed for low trade frequency (12-37/year on 6h) to minimize fee drag
# Works in bull (price above 1w EMA with bullish Alligator) and bear (price below 1w EMA with bearish Alligator) markets

name = "6h_WilliamsAlligator_1wEMA34_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA(34) trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1w for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA to 6h timeframe (wait for completed 1w bar)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Williams Alligator on 6h
    # Jaw (blue line): 13-period SMMA shifted 8 bars ahead
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth (red line): 8-period SMMA shifted 5 bars ahead
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips (green line): 5-period SMMA shifted 3 bars ahead
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(13, 8, 5, 20) + 8  # Alligator max shift(8), volume MA(20)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or
            np.isnan(lips[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Alligator bullish (jaw < teeth < lips), price > 1w EMA34, volume spike
            if (jaw[i] < teeth[i] < lips[i] and 
                close[i] > ema_34_1w_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Alligator bearish (jaw > teeth > lips), price < 1w EMA34, volume spike
            elif (jaw[i] > teeth[i] > lips[i] and 
                  close[i] < ema_34_1w_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator turns bearish or price < 1w EMA34
            if (jaw[i] >= teeth[i] or teeth[i] >= lips[i] or 
                close[i] < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator turns bullish or price > 1w EMA34
            if (jaw[i] <= teeth[i] or teeth[i] <= lips[i] or 
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals