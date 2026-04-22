#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator trend filter with 1d ATR-based entry levels and volume confirmation.
# Alligator uses SMAs (13,8,5) with 8,5,3 period offsets to identify trends.
# Long when price > Alligator Teeth (SMA8) + price > 1d ATR-based upper band + volume spike.
# Short when price < Alligator Teeth (SMA8) + price < 1d ATR-based lower band + volume spike.
# Exit when price crosses Alligator Jaw (SMA13) or ATR-based middle band.
# Designed for 4h to capture trends with minimal whipsaw in both bull/bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for ATR and Alligator components (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14)
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, tr2)
    tr = np.concatenate([[np.nan], tr])  # align length
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Alligator components from 1d close
    # Jaw: SMA(13) offset by 8 bars
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    jaw = np.concatenate([np.full(8, np.nan), jaw[:-8]])  # shift right by 8
    # Teeth: SMA(8) offset by 5 bars
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    teeth = np.concatenate([np.full(5, np.nan), teeth[:-5]])  # shift right by 5
    # Lips: SMA(5) offset by 3 bars
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    lips = np.concatenate([np.full(3, np.nan), lips[:-3]])  # shift right by 3
    
    # 1d ATR-based bands (middle = close, upper/lower = close ± ATR*1.5)
    atr_mult = 1.5
    upper_band = close_1d + atr_14_1d * atr_mult
    lower_band = close_1d - atr_14_1d * atr_mult
    middle_band = close_1d
    
    # Align to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    middle_aligned = align_htf_to_ltf(prices, df_1d, middle_band)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Higher threshold for fewer trades
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(teeth_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or
            np.isnan(middle_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > Teeth + price > Upper band + volume spike
            if (close[i] > teeth_aligned[i] and 
                close[i] > upper_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price < Teeth + price < Lower band + volume spike
            elif (close[i] < teeth_aligned[i] and 
                  close[i] < lower_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit on price < Jaw or price < Middle band
                if (close[i] < jaw_aligned[i] or close[i] < middle_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on price > Jaw or price > Middle band
                if (close[i] > jaw_aligned[i] or close[i] > middle_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_TeethFilter_1dATR_Bands_VolumeSpike"
timeframe = "4h"
leverage = 1.0