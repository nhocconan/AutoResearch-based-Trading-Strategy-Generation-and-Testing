#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Elder Ray + 12h Trend + Volume Spike
# Hypothesis: Elder Ray (bull/bear power) identifies institutional buying/selling pressure.
# Combined with 12h trend filter and volume spikes, captures strong moves in both bull and bear markets.
# Works in bull markets via buy power + uptrend, in bear via sell power + downtrend.
# Volume spikes confirm institutional participation.
# Target: 12-37 trades/year (50-150 total over 4 years) for 6h timeframe.

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
    
    # Elder Ray components (13-period EMA standard)
    ema13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema13
    # Bear Power = Low - EMA13
    bear_power = low - ema13
    
    # Smooth the power signals (13-period EMA of raw power)
    bull_power_smooth = pd.Series(bull_power).ewm(span=13, adjust=False).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=13, adjust=False).mean().values
    
    # Volume confirmation: volume > 2x 20-period average (higher threshold for fewer trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(ema_20_12h_aligned[i]) or np.isnan(bull_power_smooth[i]) or 
            np.isnan(bear_power_smooth[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: bear power turns positive (selling pressure gone) or trend turns bearish
            if bear_power_smooth[i] > 0 or close[i] < ema_20_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: bull power turns negative (buying pressure gone) or trend turns bullish
            if bull_power_smooth[i] < 0 or close[i] > ema_20_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Strong buying pressure + uptrend
                if bull_power_smooth[i] > 0 and close[i] > ema_20_12h_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Strong selling pressure + downtrend
                elif bear_power_smooth[i] < 0 and close[i] < ema_20_12h_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals