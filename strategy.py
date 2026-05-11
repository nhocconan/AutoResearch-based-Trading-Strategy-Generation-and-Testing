#!/usr/bin/env python3
name = "1d_Williams_Alligator_ElderRay_Trend_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def williams_alligator(high, low, close):
    # Williams Alligator: Smoothed Moving Average (SMMA)
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    # Lips: 5-period SMMA of median price, shifted 3 bars
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8)
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5)
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3)
    return jaw.values, teeth.values, lips.values

def elder_ray(high, low, close, ema_period=13):
    # Elder Ray: Bull Power = High - EMA, Bear Power = Low - EMA
    ema = pd.Series(close).ewm(span=ema_period, min_periods=ema_period, adjust=False).mean()
    bull_power = high - ema.values
    bear_power = low - ema.values
    return bull_power, bear_power, ema.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly trend filter (HTF) - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    weekly_ema50 = pd.Series(weekly_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema50)
    
    # Williams Alligator on daily - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    jaw, teeth, lips = williams_alligator(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Elder Ray on daily - ONCE before loop
    bull_power, bear_power, elder_ema = elder_ray(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    elder_ema_aligned = align_htf_to_ltf(prices, df_1d, elder_ema)
    
    # Volume filter: 20-period EMA for volume spike
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    # Fixed position size
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(weekly_ema50_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(elder_ema_aligned[i]) or np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Williams Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        alligator_long = lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]
        alligator_short = lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]
        
        # Elder Ray: Bull Power > 0 and Bear Power < 0 for strong trend
        elder_long = bull_power_aligned[i] > 0 and bear_power_aligned[i] < 0
        elder_short = bull_power_aligned[i] < 0 and bear_power_aligned[i] > 0
        
        # Weekly trend filter
        weekly_uptrend = close[i] > weekly_ema50_aligned[i]
        weekly_downtrend = close[i] < weekly_ema50_aligned[i]
        
        if position == 0:
            # Long: Alligator uptrend + Elder Ray bullish + weekly uptrend + volume spike
            if alligator_long and elder_long and weekly_uptrend and volume_ok[i]:
                signals[i] = position_size
                position = 1
            # Short: Alligator downtrend + Elder Ray bearish + weekly downtrend + volume spike
            elif alligator_short and elder_short and weekly_downtrend and volume_ok[i]:
                signals[i] = -position_size
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit: Alligator reverses OR Elder Ray weakens OR weekly trend fails
                if not (alligator_long and elder_long and weekly_uptrend):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: Alligator reverses OR Elder Ray weakens OR weekly trend fails
                if not (alligator_short and elder_short and weekly_downtrend):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals