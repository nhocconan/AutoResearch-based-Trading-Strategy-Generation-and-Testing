#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d trend filter and volume confirmation.
# Uses Williams Alligator (Jaw/Teeth/Lips) to identify trend absence (all lines intertwined = chop) 
# vs trend presence (lines separated and ordered). Entry when price breaks out of Alligator's 
# "mouth" in direction of 1d EMA50 trend with volume confirmation (>1.3x 20-bar avg). 
# Exit when price re-enters Alligator's mouth or volume drops. 
# Session filter (08-20 UTC) to trade only during liquid hours.
# Discrete position sizing at ±0.25 to manage fee drag.
# Target: 80-160 total trades over 4 years (20-40/year) to balance opportunity and fees on 4h timeframe.
# Works in bull markets via trend continuation and in bear markets via volatility expansion capture 
# during strong moves that break Alligator equilibrium.

name = "4h_WilliamsAlligator_1dEMA50_Trend_VolumeConfirm_Session_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d_vals = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d_vals).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 4h: SMAs of median price (typical price)
    # Jaw: 13-period SMMA shifted 8 bars forward
    # Teeth: 8-period SMMA shifted 5 bars forward  
    # Lips: 5-period SMMA shifted 3 bars forward
    # Using close as approximation for typical price (hlc3) for simplicity
    ma_13 = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    ma_8 = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    ma_5 = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    # Apply forward shifts (SMMA would use different calculation, but SMA with shift approximates)
    jaw = np.roll(ma_13, 8)  # shifted 8 bars forward
    teeth = np.roll(ma_8, 5)  # shifted 5 bars forward
    lips = np.roll(ma_5, 3)   # shifted 3 bars forward
    # Fill shifted values with NaN for lookback period
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Alligator's mouth: average of highest (lips) and lowest (jaw) when trending
    # In chop, all lines are close together
    alligator_high = np.maximum.reduce([jaw, teeth, lips])
    alligator_low = np.minimum.reduce([jaw, teeth, lips])
    alligator_mid = (alligator_high + alligator_low) / 2
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for Alligator lines and EMA50
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(volume_confirm[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        curr_volume_confirm = volume_confirm[i]
        curr_alligator_high = alligator_high[i]
        curr_alligator_low = alligator_low[i]
        
        if position == 0:  # Flat - look for new entries
            # Check if Alligator is "sleeping" (lines intertwined) or "awakening" (separating)
            # Lines are intertwined when max-min < 0.5% of price (chop condition)
            alligator_range = curr_alligator_high - curr_alligator_low
            is_chop = alligator_range < (curr_close * 0.005)
            
            # Long: price breaks above Alligator's mouth, above 1d EMA50, not deep chop, volume spike
            if (curr_close > curr_alligator_high and 
                curr_close > curr_ema_50_1d and 
                not is_chop and  # Avoid entries in deep chop
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Alligator's mouth, below 1d EMA50, not deep chop, volume spike
            elif (curr_close < curr_alligator_low and 
                  curr_close < curr_ema_50_1d and 
                  not is_chop and  # Avoid entries in deep chop
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price re-enters Alligator's mouth or strong reversal signs
            if curr_close < curr_alligator_mid:  # Price back below Alligator midpoint
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price re-enters Alligator's mouth
            if curr_close > curr_alligator_mid:  # Price back above Alligator midpoint
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals