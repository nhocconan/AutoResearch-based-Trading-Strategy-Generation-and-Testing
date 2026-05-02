#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray combo with 1d EMA34 trend filter and volume confirmation
# Uses Williams Alligator (jaw/teeth/lips) to identify trend absence/presence, Elder Ray (bull/bear power) for momentum.
# 1d EMA34 provides robust trend filter to avoid counter-trend trades. Volume spike (2.0x 20-period average) ensures participation.
# Designed for very low trade frequency (<50 total trades) to minimize fee drag while maintaining edge in ranging and trending markets.
# Works in bull markets via trend-following signals, in bear via mean-reversion from extreme Elder Ray readings.

name = "12h_WilliamsAlligator_ElderRay_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Williams Alligator: SMA of median price (typical price)
    # Jaw: 13-period SMA, shifted 8 bars
    # Teeth: 8-period SMA, shifted 5 bars
    # Lips: 5-period SMA, shifted 3 bars
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Alligator and EMA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Alligator sleeping: intertwined lines (no trend) -> look for reversal from extreme Elder Ray
            alligator_sleeping = (abs(jaw[i] - teeth[i]) < (close[i] * 0.001)) and (abs(teeth[i] - lips[i]) < (close[i] * 0.001))
            
            # Long: Extreme bear power (oversold) + price > 1d EMA34 + volume spike
            if bear_power[i] < -np.std(bull_power[max(0, i-50):i]) * 2.0 and close[i] > ema_34_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Extreme bull power (overbought) + price < 1d EMA34 + volume spike
            elif bull_power[i] > np.std(bear_power[max(0, i-50):i]) * 2.0 and close[i] < ema_34_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator awakening with bearish alignment OR extreme bull power
            alligator_bearish = lips[i] < teeth[i] < jaw[i]  # Perfect bearish alignment
            extreme_bull = bull_power[i] > np.std(bull_power[max(0, i-50):i]) * 3.0
            
            if alligator_bearish or extreme_bull:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator awakening with bullish alignment OR extreme bear power
            alligator_bullish = jaw[i] < teeth[i] < lips[i]  # Perfect bullish alignment
            extreme_bear = bear_power[i] < -np.std(bear_power[max(0, i-50):i]) * 3.0
            
            if alligator_bullish or extreme_bear:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals