#!/usr/bin/env python3
# 6h_elder_ray_regime_volume_v1
# Hypothesis: 6h strategy using Elder Ray Index (Bull/Bear Power) with 12h EMA trend filter and volume confirmation.
# In trending markets, Elder Ray shows persistence of bull/bear power aligned with trend.
# Volume confirms conviction. Uses discrete sizing (0.0, ±0.25) to minimize fee churn.
# Target: 50-150 total trades over 4 years by requiring Elder Ray extreme + EMA alignment + volume spike.
# Primary timeframe: 6h, HTF: 12h for EMA trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_elder_ray_regime_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h HTF data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # 12h EMA(20) for trend direction
    ema_12h = pd.Series(close_12h).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Elder Ray components (13-period EMA for reference)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema13
    # Bear Power = Low - EMA13
    bear_power = low - ema13
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(volume_ma[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.8x 20-period average
        volume_confirmed = volume[i] > 1.8 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: Bear Power turns positive (selling pressure) or volume dries up
            if bear_power[i] > 0 or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bull Power turns negative (buying pressure) or volume dries up
            if bull_power[i] < 0 or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                # Long entry: Bull Power strongly positive AND price above 12h EMA (uptrend)
                if bull_power[i] > 0.5 * np.std(bull_power[max(0, i-50):i]) and close[i] > ema_12h_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: Bear Power strongly negative AND price below 12h EMA (downtrend)
                elif bear_power[i] < -0.5 * np.std(bear_power[max(0, i-50):i]) and close[i] < ema_12h_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals