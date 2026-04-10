#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume confirmation and chop regime filter
# - Donchian(20) on 4h: breakout above upper band = long signal, below lower band = short signal
# - 1d volume spike: current volume > 1.8x 20-period average for confirmation
# - 4h chop regime: CHOP(14) between 38.2 and 61.8 = ranging market (avoid breakouts)
# - Long: price breaks above Donchian(20) upper band AND 1d volume spike AND chop < 38.2 (trending)
# - Short: price breaks below Donchian(20) lower band AND 1d volume spike AND chop < 38.2 (trending)
# - Exit: price returns to Donchian(20) midpoint OR chop > 61.8 (strong ranging)
# - Target: 20-50 trades/year on 4h (80-200 total over 4 years) to avoid fee drag

name = "4h_1d_donchian_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Pre-compute 4h Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Pre-compute 4h Chopiness Index (14-period)
    atr_period = 14
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    highest_high_atr = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low_atr = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    
    chop_raw = np.where((highest_high_atr - lowest_low_atr) != 0,
                        100 * np.log10(atr * atr_period / (highest_high_atr - lowest_low_atr)) / np.log10(atr_period),
                        50.0)  # neutral when range=0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(chop_raw[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition: current 1d volume > 1.8x 20-period average
        volume_spike = df_1d['volume'].values[i] > 1.8 * vol_ma_20_1d[i]
        
        # Chop regime: < 38.2 = trending, > 61.8 = ranging, 38.2-61.8 = transition
        chop_now = chop_raw[i]
        trending_regime = chop_now < 38.2
        ranging_regime = chop_now > 61.8
        
        # Breakout conditions
        long_breakout = close[i] > donchian_upper[i]
        short_breakout = close[i] < donchian_lower[i]
        
        # Return to midpoint
        return_to_mid = np.abs(close[i] - donchian_mid[i]) < 0.1 * (donchian_upper[i] - donchian_lower[i])
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: breakout above upper band AND volume spike AND trending regime
            if long_breakout and volume_spike and trending_regime:
                position = 1
                signals[i] = 0.25
            # Short conditions: breakout below lower band AND volume spike AND trending regime
            elif short_breakout and volume_spike and trending_regime:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: return to midpoint OR strong ranging regime
            exit_condition = return_to_mid or ranging_regime
            
            if exit_condition:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals