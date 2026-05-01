#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray + volume confirmation.
# Uses Williams Alligator (Jaw/Teeth/Lips) for trend direction, Elder Ray (Bull/Bear Power) for momentum,
# and volume > 1.5x 20-period median for confirmation. Works in bull (buy when Bull Power > 0 and price > Teeth)
# and bear (sell when Bear Power < 0 and price < Teeth). Discrete position sizing (0.25) to minimize fee churn.
# Target: 50-150 total trades over 4 years (12-37/year).

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
    
    # Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs of median price
    median_price_1w = (df_1w['high'] + df_1w['low']) / 2
    jaw = pd.Series(median_price_1w).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(median_price_1w).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(median_price_1w).rolling(window=5, min_periods=5).mean().values
    
    # Align Alligator components to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # Load 1d data ONCE before loop for Elder Ray (13-period EMA of high/low)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Elder Ray: Bull Power = High - EMA13(Close), Bear Power = Low - EMA13(Close)
    ema_13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = df_1d['high'].values - ema_13_1d
    bear_power = df_1d['low'].values - ema_13_1d
    
    # Align Elder Ray components to 12h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for Alligator (13), Elder Ray (13), volume median (20)
    start_idx = 20
    
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
        
        # Williams Alligator trend: price > Teeth = uptrend, price < Teeth = downtrend
        uptrend = curr_close > teeth_aligned[i]
        downtrend = curr_close < teeth_aligned[i]
        
        # Elder Ray momentum: Bull Power > 0 = bullish, Bear Power < 0 = bearish
        bullish_momentum = bull_power_aligned[i] > 0
        bearish_momentum = bear_power_aligned[i] < 0
        
        # Volume confirmation: current volume > 1.5x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.5)
        
        if position == 0:  # Flat - look for new entries
            # Long: Uptrend AND bullish momentum AND volume confirmation
            if uptrend and bullish_momentum and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Downtrend AND bearish momentum AND volume confirmation
            elif downtrend and bearish_momentum and volume_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on downtrend or bearish momentum
            if not (uptrend and bullish_momentum):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on uptrend or bullish momentum
            if not (downtrend and bearish_momentum):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals