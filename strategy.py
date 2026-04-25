#!/usr/bin/env python3
"""
4h Williams Alligator + 1d EMA34 Trend + Volume Spike
Hypothesis: Williams Alligator (JAW/TEETH/LIPS) identifies trendless markets; when all three lines are aligned and price breaks out with volume, it signals strong trend continuation. 1d EMA34 filters higher-timeframe trend direction. Works in bull/bear by only taking breakouts in the direction of the 1d trend.
Target: 20-50 trades/year (80-200 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator on 4h: SMAs of median price
    median_price = (high + low) / 2.0
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values  # 13-period, shifted 8
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values   # 8-period, shifted 5
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values    # 5-period, shifted 3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Alligator (max shift 8) + EMA34 warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(ema_34_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_trend = ema_34_aligned[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Alligator alignment: check if lines are ordered (trending) or tangled (ranging)
        # Bullish alignment: Lips > Teeth > Jaw
        # Bearish alignment: Lips < Teeth < Jaw
        bullish_aligned = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        bearish_aligned = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
        
        if position == 0:
            # Long: Bullish aligned + price above Lips (breakout) + above 1d EMA34 (uptrend) + volume spike
            long_condition = bullish_aligned and (curr_close > lips[i]) and (curr_close > ema_trend) and volume_spike
            # Short: Bearish aligned + price below Lips (breakdown) + below 1d EMA34 (downtrend) + volume spike
            short_condition = bearish_aligned and (curr_close < lips[i]) and (curr_close < ema_trend) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Teeth or trend breaks
            if curr_close < teeth[i] or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Teeth or trend breaks
            if curr_close > teeth[i] or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0