#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Elder Ray + 12h Trend + Volume Spike
# Hypothesis: Elder Ray (bull/bear power) captures institutional buying/selling pressure
# Combined with 12h EMA trend filter and volume spike confirmation to enter high-probability
# moves in both bull and bear markets. Exit when power fades or trend weakens.
# Target: 50-120 total trades over 4 years (12-30/year).

name = "6h_elder_ray_12h_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # 12h EMA(34) for trend filter (Fibonacci number)
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    # Using 13-period EMA on 6h data (standard for Elder Ray)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    # Volume confirmation: volume > 2x 30-period average (strict for low frequency)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=15).mean().values
    vol_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: bull power weakens OR trend turns bearish
            if bull_power[i] <= 0 or close[i] < ema_34_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: bear power weakens OR trend turns bullish
            if bear_power[i] <= 0 or close[i] > ema_34_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Strong bull power with uptrend
                if bull_power[i] > 0 and close[i] > ema_34_12h_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Strong bear power with downtrend
                elif bear_power[i] > 0 and close[i] < ema_34_12h_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals