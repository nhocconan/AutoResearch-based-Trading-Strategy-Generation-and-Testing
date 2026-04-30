#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA34 trend filter and volume confirmation
# Williams Alligator identifies trend via three SMAs (Jaw/Teeth/Lips). 
# When all three align (Jaw > Teeth > Lips for uptrend, reverse for downtrend) + price confirms,
# it signals strong trend. Volume confirmation ensures legitimacy. Works in bull/bear:
# - Bull: Alligator aligned up + price above Lips + volume spike
# - Bear: Alligator aligned down + price below Lips + volume spike
# Uses 12h timeframe to target 50-150 trades over 4 years (12-37/year), avoiding fee drag.
# Discrete sizing (0.25) minimizes churn.

name = "12h_WilliamsAlligator_1dEMA34_Trend_Volume_v1"
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
    
    # Williams Alligator (13,8,5 SMAs with offsets)
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    jaw_offset = 8
    teeth_offset = 5
    lips_offset = 3
    
    close_s = pd.Series(close)
    jaw = close_s.rolling(window=jaw_period, min_periods=jaw_period).mean().shift(jaw_offset).values
    teeth = close_s.rolling(window=teeth_period, min_periods=teeth_period).mean().shift(teeth_offset).values
    lips = close_s.rolling(window=lips_period, min_periods=lips_period).mean().shift(lips_offset).values
    
    # Alligator alignment conditions
    alligator_up = (jaw > teeth) & (teeth > lips)  # Uptrend alignment
    alligator_down = (jaw < teeth) & (teeth < lips)  # Downtrend alignment
    
    # Price relative to Lips (trigger line)
    price_above_lips = close > lips
    price_below_lips = close < lips
    
    # Volume confirmation: volume > 2.0x 20-period average (stricter for 12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, jaw_period + jaw_offset, teeth_period + teeth_offset, lips_period + lips_offset, 20, 34)
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_alligator_up = alligator_up[i]
        curr_alligator_down = alligator_down[i]
        curr_price_above_lips = price_above_lips[i]
        curr_price_below_lips = price_below_lips[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade after Alligator alignment with price confirmation, volume, and trend filter
            if curr_volume_confirm:
                # Bullish: Alligator up + price above lips + above 1d EMA34
                if curr_alligator_up and curr_price_above_lips and curr_close > curr_ema_34_1d:
                    signals[i] = 0.25
                    position = 1
                # Bearish: Alligator down + price below lips + below 1d EMA34
                elif curr_alligator_down and curr_price_below_lips and curr_close < curr_ema_34_1d:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: Alligator realigns down OR price closes below lips
            if curr_alligator_down or curr_close < lips[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator realigns up OR price closes above lips
            if curr_alligator_up or curr_close > lips[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals