#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d volume spike + chop regime filter
# - Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) SMAs on median price
# - Long when Lips > Teeth > Jaw (bullish alignment) AND volume > 2x 20-period average AND CHOP > 61.8 (ranging)
# - Short when Lips < Teeth < Jaw (bearish alignment) AND volume > 2x 20-period average AND CHOP > 61.8 (ranging)
# - Exit when Alligator alignment breaks OR CHOP < 38.2 (trending)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Alligator identifies trend via smoothed median price
# - Volume confirmation reduces false signals
# - Chop filter ensures we trade in ranging markets where mean reversion works

name = "12h_1d_alligator_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 12h median price for Alligator
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    median_price = (high + low) / 2
    
    # Pre-compute 12h Williams Alligator
    def sma(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).mean().values
    
    jaw = sma(median_price, 13)  # Jaw: 13-period, 8-shift
    teeth = sma(median_price, 8)  # Teeth: 8-period, 5-shift
    lips = sma(median_price, 5)   # Lips: 5-period, 3-shift
    
    # Apply shifts (Alligator uses future-shifted SMAs)
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # Set NaN for shifted values
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Alligator alignment
    lips_above_teeth = lips > teeth
    teeth_above_jaw = teeth > jaw
    lips_below_teeth = lips < teeth
    teeth_below_jaw = teeth < jaw
    bullish_alignment = lips_above_teeth & teeth_above_jaw
    bearish_alignment = lips_below_teeth & teeth_below_jaw
    
    # Pre-compute 1d volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Pre-compute 12h Choppiness Index (CHOP)
    def true_range(high, low, prev_close):
        tr1 = high - low
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    prev_close = np.roll(close, 1)
    prev_close[0] = np.nan
    tr = true_range(high, low, prev_close)
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    denominator = max_high - min_low
    chop = np.where(denominator != 0, 
                    100 * np.log10(atr14 * np.sqrt(14) / denominator) / np.log10(10),
                    50)  # neutral when range is zero
    
    chop_high = chop > 61.8  # ranging market
    chop_low = chop < 38.2   # trending market
    
    # Align HTF indicators to 12h timeframe
    bullish_alignment_aligned = align_htf_to_ltf(prices, df_1d, bullish_alignment)
    bearish_alignment_aligned = align_htf_to_ltf(prices, df_1d, bearish_alignment)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    chop_high_aligned = align_htf_to_ltf(prices, df_1d, chop_high)
    chop_low_aligned = align_htf_to_ltf(prices, df_1d, chop_low)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(bullish_alignment_aligned[i]) or np.isnan(bearish_alignment_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(chop_high_aligned[i]) or 
            np.isnan(chop_low_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: bullish Alligator alignment AND volume spike AND chop > 61.8 (ranging)
            if (bullish_alignment_aligned[i] and 
                volume_spike_aligned[i] and 
                chop_high_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: bearish Alligator alignment AND volume spike AND chop > 61.8 (ranging)
            elif (bearish_alignment_aligned[i] and 
                  volume_spike_aligned[i] and 
                  chop_high_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: Alligator alignment breaks OR chop < 38.2 (trending)
            exit_long = (position == 1 and 
                        (not bullish_alignment_aligned[i] or chop_low_aligned[i]))
            exit_short = (position == -1 and 
                         (not bearish_alignment_aligned[i] or chop_low_aligned[i]))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals