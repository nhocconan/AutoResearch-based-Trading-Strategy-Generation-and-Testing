#!/usr/bin/env python3
name = "1d_WilliamsAlligator_ElderRay_Trend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator: SMA of median price
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8)  # 13-period SMA, shifted 8
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5)   # 8-period SMA, shifted 5
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3)    # 5-period SMA, shifted 3
    
    # Elder Ray: Bull/Bear Power
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # 1-Week Trend (HTF)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Daily Volume Spike
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg_20)
    
    # Hour filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bullish alignment (Lips > Teeth > Jaw) + Bull Power > 0 + Above weekly EMA34 + Volume spike
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and
                bull_power[i] > 0 and
                close[i] > ema34_1w_aligned[i] and
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment (Lips < Teeth < Jaw) + Bear Power < 0 + Below weekly EMA34 + Volume spike
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and
                  bear_power[i] < 0 and
                  close[i] < ema34_1w_aligned[i] and
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator turns bearish OR Bull Power <= 0 OR closes below weekly EMA34
            if (lips[i] <= teeth[i] or teeth[i] <= jaw[i] or
                bull_power[i] <= 0 or
                close[i] <= ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator turns bullish OR Bear Power >= 0 OR closes above weekly EMA34
            if (lips[i] >= teeth[i] or teeth[i] >= jaw[i] or
                bear_power[i] >= 0 or
                close[i] >= ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals