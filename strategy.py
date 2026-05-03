#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA(34) trend filter and volume confirmation
# Elder Ray measures bull/bear power relative to EMA: Bull Power = High - EMA, Bear Power = Low - EMA
# Long when Bull Power > 0 and rising, price above 1d EMA34, volume > 1.5x 20-period EMA
# Short when Bear Power < 0 and falling, price below 1d EMA34, volume > 1.5x 20-period EMA
# Works in bull/bear markets by aligning with 1d trend while capturing momentum shifts
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

name = "6h_ElderRay_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6h EMA(13) for Elder Ray (standard period)
    close_s = pd.Series(close)
    ema_13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA
    bear_power = low - ema_13   # Bear Power = Low - EMA
    
    # Volume confirmation: 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.5 x 20-period EMA (balanced to avoid overtrading)
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        # Elder Ray signals with 1d trend filter
        # Long: Rising Bull Power (>0) + price above 1d EMA34 + volume spike
        # Short: Falling Bear Power (<0) + price below 1d EMA34 + volume spike
        if position == 0:
            if i > 50:  # Need previous bar for momentum
                bull_rising = bull_power[i] > bull_power[i-1]
                bear_falling = bear_power[i] < bear_power[i-1]
                
                if bull_power[i] > 0 and bull_rising and close[i] > ema_34_1d_aligned[i] and volume_spike:
                    signals[i] = 0.25
                    position = 1
                elif bear_power[i] < 0 and bear_falling and close[i] < ema_34_1d_aligned[i] and volume_spike:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: Bull Power turns negative OR price breaks below 1d EMA34
            if bull_power[i] <= 0 or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power turns positive OR price breaks above 1d EMA34
            if bear_power[i] >= 0 or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals