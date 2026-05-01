#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume spike and ATR-based regime filter.
# Long when price breaks above Donchian upper band with volume > 2x median and ATR(14) > ATR(50) (volatile regime).
# Short when price breaks below Donchian lower band with volume > 2x median and ATR(14) > ATR(50).
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 75-200 total trades over 4 years.
# Works in bull (buy breakouts with volatility expansion) and bear (sell breakdowns with volatility expansion).

name = "4h_Donchian20_Breakout_VolumeSpike_ATRRegime_v1"
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
    
    # Calculate Donchian channels (20-period)
    donchian_h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_l = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) and ATR(50) for regime filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for ATR50 and Donchian
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_h[i]) or 
            np.isnan(donchian_l[i]) or
            np.isnan(atr_14[i]) or
            np.isnan(atr_50[i]) or
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Regime filter: ATR(14) > ATR(50) (volatile market regime)
        volatile_regime = atr_14[i] > atr_50[i]
        
        # Volume confirmation: current volume > 2.0x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 2.0)
        
        # Donchian breakout conditions
        breakout_up = curr_close > donchian_h[i]   # break above upper band
        breakout_down = curr_close < donchian_l[i] # break below lower band
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout up AND volatile regime AND volume confirmation
            if breakout_up and volatile_regime and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Breakout down AND volatile regime AND volume confirmation
            elif breakout_down and volatile_regime and volume_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Donchian breakout down (reversal signal)
            if breakout_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Donchian breakout up (reversal signal)
            if breakout_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals