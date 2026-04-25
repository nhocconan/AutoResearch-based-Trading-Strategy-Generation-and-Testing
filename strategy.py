#!/usr/bin/env python3
"""
1d_Alligator_JawTeethLips_1wTrend_VolumeConfirm
Hypothesis: Williams Alligator (Jaw=EMA13(8), Teeth=EMA8(5), Lips=EMA5(3)) on 1d with 1w trend filter (price >/< EMA34 weekly) and volume confirmation (>1.5x 20-bar avg). 
Enters long when Lips > Teeth > Jaw (bullish alignment) in 1d uptrend with volume spike, short when Lips < Teeth < Jaw (bearish alignment) in 1d downtrend with volume spike. 
Exits on opposite Alligator alignment or trend reversal. 
Designed for 1d timeframe with ~10-25 trades/year, works in bull/bear by following 1w trend filter and Alligator alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Williams Alligator on 1d timeframe
    # Jaw: EMA13 of median price, shifted 8 bars
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).ewm(span=13, adjust=False, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)  # shift 8 bars forward
    jaw[:8] = np.nan
    
    # Teeth: EMA8 of median price, shifted 5 bars
    teeth = pd.Series(median_price).ewm(span=8, adjust=False, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)  # shift 5 bars forward
    teeth[:5] = np.nan
    
    # Lips: EMA5 of median price, shifted 3 bars
    lips = pd.Series(median_price).ewm(span=5, adjust=False, min_periods=5).mean().values
    lips = np.roll(lips, 3)  # shift 3 bars forward
    lips[:3] = np.nan
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need at least 1 bar of previous data and Alligator warmup
    start_idx = max(13, 8, 5, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(jaw[i]) or 
            np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) in 1d uptrend with volume confirmation
            bullish_alignment = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
            long_setup = bullish_alignment and (close[i] > ema_34_1w_aligned[i]) and volume_spike[i]
            # Short: Lips < Teeth < Jaw (bearish alignment) in 1d downtrend with volume confirmation
            bearish_alignment = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
            short_setup = bearish_alignment and (close[i] < ema_34_1w_aligned[i]) and volume_spike[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: bearish Alligator alignment OR trend turns down
            bearish_alignment = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
            if bearish_alignment or (close[i] < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: bullish Alligator alignment OR trend turns up
            bullish_alignment = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
            if bullish_alignment or (close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Alligator_JawTeethLips_1wTrend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0