#!/usr/bin/env python3
"""
6h Elder Ray Power + 1d EMA34 Trend + Volume Spike
Hypothesis: Elder Ray Bull Power (high-EMA13) and Bear Power (low-EMA13) measure buying/selling pressure.
In strong trends (1d EMA34), extreme power readings with volume spike indicate exhaustion and reversal.
Works in bull/bear via trend filter: long when bear power extreme + uptrend, short when bull power extreme + downtrend.
Target: 12-37 trades/year on 6h.
"""

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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate EMA13 for Elder Ray (on 6h data)
    if len(close) >= 13:
        ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    else:
        ema_13 = np.full(n, np.nan)
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # Buying pressure
    bear_power = low - ema_13   # Selling pressure
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for data to propagate
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_34 = ema_34_1d_aligned[i]
        bp = bull_power[i]
        br = bear_power[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Trend filter
        uptrend = curr_close > ema_34
        downtrend = curr_close < ema_34
        
        if position == 0:
            # Long: extreme bear power (selling exhaustion) AND volume spike AND uptrend
            # Bear power is negative; more negative = stronger selling
            long_condition = (br < -np.std(br[max(0, i-50):i+1]) * 1.5) and volume_spike and uptrend
            # Short: extreme bull power (buying exhaustion) AND volume spike AND downtrend
            # Bull power is positive; more positive = stronger buying
            short_condition = (bp > np.std(bp[max(0, i-50):i+1]) * 1.5) and volume_spike and downtrend
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss or trend reversal or power normalization
            if (curr_close <= entry_price - 2.0 * np.std(close[max(0, i-20):i+1]) or 
                not uptrend or 
                br > -np.std(br[max(0, i-20):i+1]) * 0.5):  # selling pressure normalized
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss or trend reversal or power normalization
            if (curr_close >= entry_price + 2.0 * np.std(close[max(0, i-20):i+1]) or 
                not downtrend or 
                bp < np.std(bp[max(0, i-20):i+1]) * 0.5):  # buying pressure normalized
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Power_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0