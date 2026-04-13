#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull Power/Bear Power) with 1-week trend filter and volume confirmation.
# Elder Ray measures bull/bear power relative to EMA. Bull Power = High - EMA, Bear Power = Low - EMA.
# In strong uptrends: Bull Power > 0 and rising, Bear Power < 0. In downtrends: Bear Power < 0 and falling, Bull Power > 0.
# Combined with 1-week EMA trend filter to avoid counter-trend trades and volume spikes for confirmation.
# Target: 15-35 trades per year (60-140 total over 4 years) for 6h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # EMA(20) for 1-week trend filter
    ema20_1w = np.zeros(len(close_1w))
    ema_multiplier = 2 / (20 + 1)
    ema20_1w[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        ema20_1w[i] = (close_1w[i] - ema20_1w[i-1]) * ema_multiplier + ema20_1w[i-1]
    
    # Align 1-week EMA to 6h timeframe
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Elder Ray on 6h timeframe: EMA(13) as base
    ema13 = np.zeros(n)
    ema_multiplier_13 = 2 / (13 + 1)
    ema13[0] = close[0]
    for i in range(1, n):
        ema13[i] = (close[i] - ema13[i-1]) * ema_multiplier_13 + ema13[i-1]
    
    # Bull Power = High - EMA13
    bull_power = high - ema13
    # Bear Power = Low - EMA13
    bear_power = low - ema13
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if np.isnan(ema20_1w_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(avg_volume[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_trend = ema20_1w_aligned[i]
        bp = bull_power[i]
        bp_prev = bull_power[i-1] if i > 0 else 0
        be = bear_power[i]
        be_prev = bear_power[i-1] if i > 0 else 0
        
        # Volume confirmation: current volume > 2.0x average volume
        volume_confirm = vol > 2.0 * avg_vol
        
        if position == 0:
            # Long: Bull Power > 0 and rising + above 1w EMA20 + volume confirmation
            if (bp > 0 and bp > bp_prev and
                price > ema_trend and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: Bear Power < 0 and falling + below 1w EMA20 + volume confirmation
            elif (be < 0 and be < be_prev and
                  price < ema_trend and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Bull Power turns negative or price breaks below 1w EMA
            if (bp <= 0 or
                price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Bear Power turns positive or price breaks above 1w EMA
            if (be >= 0 or
                price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1w_ElderRay_Trend_Volume"
timeframe = "6h"
leverage = 1.0