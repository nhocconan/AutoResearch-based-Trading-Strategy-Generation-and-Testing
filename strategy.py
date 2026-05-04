#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + Elder Ray combination with 1d trend filter
# Uses 4h Williams Alligator (jaw/teeth/lips) for trend identification and entry signals
# Elder Ray (bull/bear power) confirms momentum strength behind the move
# 1d EMA50 provides higher timeframe trend filter to avoid counter-trend trades
# Volume confirmation (1.5x 20-period EMA) ensures strong participation
# Designed to work in both bull and bear markets by following the 1d trend direction
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag on 4h timeframe

name = "4h_WilliamsAlligator_ElderRay_1dEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 4h (jaw=13, teeth=8, lips=5 SMAs of median price)
    median_price = (high + low) / 2.0
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    
    # Elder Ray on 4h (bull power = high - EMA13, bear power = low - EMA13)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume confirmation: 20-period EMA on 4h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.5 x 20-period EMA (balanced to avoid overtrading)
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        # Williams Alligator signals:
        # Jaw > Teeth > Lips = uptrend (alligator eating with mouth up)
        # Lips > Teeth > Jaw = downtrend (alligator eating with mouth down)
        alligator_long = jaw[i] > teeth[i] and teeth[i] > lips[i]
        alligator_short = lips[i] > teeth[i] and teeth[i] > jaw[i]
        
        # Elder Ray confirmation: strong bull/bear power
        strong_bull = bull_power[i] > 0 and bull_power[i] > np.mean(bull_power[max(0, i-20):i+1])
        strong_bear = bear_power[i] < 0 and abs(bear_power[i]) > np.mean(abs(bear_power[max(0, i-20):i+1]))
        
        if position == 0:
            # Long: Alligator uptrend + volume spike + strong bull power + price above 1d EMA50 (uptrend filter)
            if (alligator_long and volume_spike and strong_bull and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Alligator downtrend + volume spike + strong bear power + price below 1d EMA50 (downtrend filter)
            elif (alligator_short and volume_spike and strong_bear and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator trend reversal OR price below 1d EMA50 (trend change)
            if not alligator_long or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator trend reversal OR price above 1d EMA50 (trend change)
            if not alligator_short or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals