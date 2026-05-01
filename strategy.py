#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray with volume confirmation.
# Long when: price > Alligator Jaw (teeth > lips) AND Bull Power > 0 AND volume > 1.5x 20-period median.
# Short when: price < Alligator Jaw (teeth < lips) AND Bear Power < 0 AND volume > 1.5x 20-period median.
# Exit on opposite Alligator signal to reduce whipsaw.
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 12-30 trades/year on 12h.
# Williams Alligator identifies trend, Elder Ray measures bull/bear power behind the move.

name = "12h_WilliamsAlligator_ElderRay_VolumeConfirm_v1"
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
    
    # Load 1w data ONCE before loop for Williams Alligator (13,8,5 SMAs on median price)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator components on 1w
    # Median price = (high + low) / 2
    median_price = (df_1w['high'] + df_1w['low']) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values  # Blue line
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values    # Red line
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values    # Green line
    
    # Align Alligator to 12h timeframe (wait for completed 1w bar)
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # Load 1d data ONCE before loop for Elder Ray (13-period EMA of high/low)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate Elder Ray components on 1d
    ema_13 = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = df_1d['high'].values - ema_13  # Bull Power = High - EMA13
    bear_power = df_1d['low'].values - ema_13   # Bear Power = Low - EMA13
    
    # Align Elder Ray to 12h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for Alligator and Elder Ray
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Williams Alligator trend: Jaw > Teeth > Lips = uptrend, Jaw < Teeth < Lips = downtrend
        uptrend = jaw_aligned[i] > teeth_aligned[i] > lips_aligned[i]
        downtrend = jaw_aligned[i] < teeth_aligned[i] < lips_aligned[i]
        
        # Elder Ray power: Bull Power > 0 = bulls in control, Bear Power < 0 = bears in control
        bulls_in_control = bull_power_aligned[i] > 0
        bears_in_control = bear_power_aligned[i] < 0
        
        # Volume confirmation: current volume > 1.5x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.5)
        
        # Alligator signals: price relative to Jaw
        price_above_jaw = curr_close > jaw_aligned[i]
        price_below_jaw = curr_close < jaw_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Price above Jaw AND uptrend AND bulls in control AND volume confirmation
            if price_above_jaw and uptrend and bulls_in_control and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Price below Jaw AND downtrend AND bears in control AND volume confirmation
            elif price_below_jaw and downtrend and bears_in_control and volume_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Alligator reversal signal (price below Jaw)
            if price_below_jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Alligator reversal signal (price above Jaw)
            if price_above_jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals