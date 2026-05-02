#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + Elder Ray combination with 1w EMA34 trend filter and volume confirmation
# Uses Williams Alligator (jaw/teeth/lips) to identify trend absence/presence, Elder Ray (bull/bear power) for momentum,
# 1w EMA34 for higher timeframe trend filter, and volume spike (2.0x 20-period average) for institutional confirmation.
# Designed for low trade frequency (<50 total trades) to minimize fee drag while maintaining edge in both bull and bear markets.
# Alligator identifies ranging markets (avoid trades), Elder Ray provides entry signals in trending conditions.

name = "1d_WilliamsAlligator_ElderRay_1wEMA34_Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for HTF calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # Williams Alligator: Smoothed Medians (5, 8, 13 periods) -> 3, 5, 8 shifted
    jaw_period = 13  # Blue line
    teeth_period = 8  # Red line
    lips_period = 5   # Green line
    jaw_shift = 8
    teeth_shift = 5
    lips_shift = 3
    
    # Calculate median prices
    median_price = (high + low) / 2
    
    # Jaw (Blue) - 13-period SMMA shifted 8 bars
    jaw = pd.Series(median_price).rolling(window=jaw_period, min_periods=jaw_period).mean().shift(jaw_shift).values
    # Teeth (Red) - 8-period SMMA shifted 5 bars
    teeth = pd.Series(median_price).rolling(window=teeth_period, min_periods=teeth_period).mean().shift(teeth_shift).values
    # Lips (Green) - 5-period SMMA shifted 3 bars
    lips = pd.Series(median_price).rolling(window=lips_period, min_periods=lips_period).mean().shift(lips_shift).values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Alligator and EMA calculations)
    start_idx = max(jaw_period + jaw_shift, teeth_period + teeth_shift, lips_period + lips_shift, 34) + 5
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Alligator sleeping condition: all lines intertwined (market ranging)
            # Avoid trading when Alligator is sleeping (no clear trend)
            alligator_sleeping = (abs(jaw[i] - teeth[i]) < (teeth[i] * 0.001) and 
                                abs(teeth[i] - lips[i]) < (lips[i] * 0.001) and
                                abs(lips[i] - jaw[i]) < (jaw[i] * 0.001))
            
            if not alligator_sleeping:
                # Long: Bull Power > 0 (bulls in control) + price > teeth + price > 1w EMA34 + volume spike
                if (bull_power[i] > 0 and close[i] > teeth[i] and close[i] > ema_34_aligned[i] and volume_spike[i]):
                    signals[i] = 0.25
                    position = 1
                # Short: Bear Power < 0 (bears in control) + price < teeth + price < 1w EMA34 + volume spike
                elif (bear_power[i] < 0 and close[i] < teeth[i] and close[i] < ema_34_aligned[i] and volume_spike[i]):
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid trading in ranging markets
        
        elif position == 1:  # Long position
            # Exit: Bear Power < 0 (bears take over) OR price breaks below lips (Alligator wake up signal)
            if bear_power[i] < 0 or close[i] < lips[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bull Power > 0 (bulls take over) OR price breaks above lips (Alligator wake up signal)
            if bull_power[i] > 0 or close[i] > lips[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals