#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w trend filter and volume confirmation
# Long when price > Alligator Jaw, Teeth > Lips, and price > 1w EMA34 + volume spike
# Short when price < Alligator Jaw, Teeth < Lips, and price < 1w EMA34 + volume spike
# Uses Williams Alligator from 12h for structure, 1w EMA for higher timeframe trend, volume for confirmation
# Designed for low trade frequency (12-37/year on 12h) to minimize fee drag
# Works in bull (price above Alligator with trend) and bear (price below Alligator with trend) markets

name = "12h_WilliamsAlligator_Volume_1wEMA34_v1"
timeframe = "12h"
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
    
    # Align 1w EMA to 12h timeframe (wait for completed 1w bar)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Williams Alligator components on 12h
    # Jaw (Blue Line): 13-period SMMA smoothed 8 periods ahead
    jaw_raw = pd.Series((high + low) / 2).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw_raw, 8)
    jaw[:8] = np.nan
    
    # Teeth (Red Line): 8-period SMMA smoothed 5 periods ahead
    teeth_raw = pd.Series((high + low) / 2).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth_raw, 5)
    teeth[:5] = np.nan
    
    # Lips (Green Line): 5-period SMMA smoothed 3 periods ahead
    lips_raw = pd.Series((high + low) / 2).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips_raw, 3)
    lips[:3] = np.nan
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(13, 8, 5, 20) + 8  # Alligator max period(13), volume MA(20), jaw shift(8)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or
            np.isnan(lips[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price > Jaw, Teeth > Lips, price > 1w EMA34, volume spike
            if (close[i] > jaw[i] and teeth[i] > lips[i] and 
                close[i] > ema_34_1w_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price < Jaw, Teeth < Lips, price < 1w EMA34, volume spike
            elif (close[i] < jaw[i] and teeth[i] < lips[i] and 
                  close[i] < ema_34_1w_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price < Jaw or Teeth < Lips or price < 1w EMA34
            if (close[i] < jaw[i] or teeth[i] < lips[i] or 
                close[i] < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price > Jaw or Teeth > Lips or price > 1w EMA34
            if (close[i] > jaw[i] or teeth[i] > lips[i] or 
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals