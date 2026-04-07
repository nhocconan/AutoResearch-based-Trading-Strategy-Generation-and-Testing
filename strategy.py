#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Elder Ray Power + 12h Trend + Volume Confirmation
# Hypothesis: Elder Ray (Bull/Bear Power) measures bull/bear strength relative to EMA.
# Combining with 12h EMA trend filter and volume spikes captures strong momentum moves
# while avoiding weak breakouts. Works in bull/bear via trend alignment.
# Target: 50-150 total trades over 4 years (12-37/year).

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
    
    # 12h EMA(20) for trend filter
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Elder Ray components (13-period EMA as base)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    bull_power = high - ema_13  # Bull Power = High - EMA(13)
    bear_power = low - ema_13   # Bear Power = Low - EMA(13)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_20_12h_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: Bear Power turns positive (selling pressure weakening) or trend turns bearish
            if bear_power[i] > 0 or close[i] < ema_20_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: Bull Power turns negative (buying pressure weakening) or trend turns bullish
            if bull_power[i] < 0 or close[i] > ema_20_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Strong bullish momentum: Bull Power > 0 AND Bear Power < 0 (both confirm uptrend)
                # Plus price above 12h EMA for trend alignment
                if bull_power[i] > 0 and bear_power[i] < 0 and close[i] > ema_20_12h_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Strong bearish momentum: Bull Power < 0 AND Bear Power > 0 (both confirm downtrend)
                # Plus price below 12h EMA for trend alignment
                elif bull_power[i] < 0 and bear_power[i] > 0 and close[i] < ema_20_12h_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals