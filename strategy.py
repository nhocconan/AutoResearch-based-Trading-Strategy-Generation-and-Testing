#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d Alligator trend filter + volume confirmation
# Elder Ray measures bull/bear power relative to EMA13. Alligator (Jaw/Teeth/Lips) provides trend direction and avoids choppy markets.
# Works in bull via Elder Ray bull power > 0 with Alligator aligned long, in bear via bear power < 0 with Alligator aligned short.
# Volume spike (2.0x 20-period average) confirms participation. Designed for 6h timeframe to target 50-150 total trades over 4 years.

name = "6h_ElderRay_Alligator_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA13, EMA8, EMA5 for Alligator (Williams Alligator)
    close_1d = df_1d['close'].values
    ema_13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values  # Jaw
    ema_8 = pd.Series(close_1d).ewm(span=8, adjust=False, min_periods=8).mean().values      # Teeth
    ema_5 = pd.Series(close_1d).ewm(span=5, adjust=False, min_periods=5).mean().values     # Lips
    
    # Align Alligator lines to 6h timeframe (wait for 1d bar close)
    ema_13_aligned = align_htf_to_ltf(prices, df_1d, ema_13)
    ema_8_aligned = align_htf_to_ltf(prices, df_1d, ema_8)
    ema_5_aligned = align_htf_to_ltf(prices, df_1d, ema_5)
    
    # Calculate Elder Ray components (need 13-period EMA for power calculation)
    ema_13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13_6h          # Bull Power = High - EMA13
    bear_power = low - ema_13_6h           # Bear Power = Low - EMA13
    
    # Calculate volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for EMA calculations)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_13_aligned[i]) or np.isnan(ema_8_aligned[i]) or np.isnan(ema_5_aligned[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Alligator aligned: Lips > Teeth > Jaw = bullish alignment
            # Alligator aligned: Jaw > Teeth > Lips = bearish alignment
            alligator_bull = ema_5_aligned[i] > ema_8_aligned[i] and ema_8_aligned[i] > ema_13_aligned[i]
            alligator_bear = ema_13_aligned[i] > ema_8_aligned[i] and ema_8_aligned[i] > ema_5_aligned[i]
            
            # Long: Bull Power > 0 (bullish momentum) + Alligator bullish alignment + volume spike
            if bull_power[i] > 0 and alligator_bull and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (bearish momentum) + Alligator bearish alignment + volume spike
            elif bear_power[i] < 0 and alligator_bear and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bear Power >= 0 (loss of bullish momentum) OR Alligator alignment breaks
            if bear_power[i] >= 0 or not (ema_5_aligned[i] > ema_8_aligned[i] and ema_8_aligned[i] > ema_13_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bull Power <= 0 (loss of bearish momentum) OR Alligator alignment breaks
            if bull_power[i] <= 0 or not (ema_13_aligned[i] > ema_8_aligned[i] and ema_8_aligned[i] > ema_5_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals