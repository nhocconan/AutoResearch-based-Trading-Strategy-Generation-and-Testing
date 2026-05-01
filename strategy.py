#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray + Volume Spike for trend following in both bull and bear markets.
# Uses 1d Williams Alligator (JAWS/TEETH/LIPS) for trend direction and 1d Elder Ray (Bull/Bear Power) for momentum confirmation.
# Long when price > Alligator JAWS AND Bull Power > 0 AND volume > 1.8x 20-bar average.
# Short when price < Alligator JAWS AND Bear Power < 0 AND volume > 1.8x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. Session filter 08-20 UTC to avoid low-liquidity hours.
# Volume threshold set to 1.8x to balance trade frequency and signal quality.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

name = "12h_WilliamsAlligator_ElderRay_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session hours for efficiency (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # Load 1d data ONCE before loop for Williams Alligator and Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d Williams Alligator: SMAs of median price
    median_price = (df_1d['high'] + df_1d['low']) / 2
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    median_price_1d = median_price.values
    
    # JAWS: 13-period SMMA shifted by 8 bars
    jaws = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    # TEETH: 8-period SMMA shifted by 5 bars
    teeth = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    # LIPS: 5-period SMMA shifted by 3 bars
    lips = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator components to 12h timeframe
    jaws_aligned = align_htf_to_ltf(prices, df_1d, jaws)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # 1d Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema_13
    bear_power = low_1d - ema_13
    
    # Align Elder Ray components to 12h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume confirmation: current 12h volume > 1.8x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Alligator and Elder Ray
    
    for i in range(start_idx, n):
        # Session filter: trade only 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.8)
        
        # Alligator trend: price above/below JAWS (13-period SMMA shifted)
        price_above_jaws = curr_close > jaws_aligned[i]
        price_below_jaws = curr_close < jaws_aligned[i]
        
        # Elder Ray momentum: Bull/Bear Power confirmation
        bull_confirm = bull_power_aligned[i] > 0
        bear_confirm = bear_power_aligned[i] < 0
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: price > JAWS AND Bull Power > 0 AND volume confirmation
            if (price_above_jaws and 
                bull_confirm and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price < JAWS AND Bear Power < 0 AND volume confirmation
            elif (price_below_jaws and 
                  bear_confirm and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below JAWS OR Bull Power <= 0 (trend/momentum change)
            if (curr_close <= jaws_aligned[i] or 
                bull_power_aligned[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above JAWS OR Bear Power >= 0 (trend/momentum change)
            if (curr_close >= jaws_aligned[i] or 
                bear_power_aligned[i] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals