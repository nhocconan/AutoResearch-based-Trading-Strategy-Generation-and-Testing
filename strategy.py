#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d EMA34 trend filter and volume confirmation
# Elder Ray = Bull Power (High - EMA13) and Bear Power (Low - EMA13)
# Long: Bull Power > 0, Bear Power < 0, price above 1d EMA34, volume spike
# Short: Bull Power < 0, Bear Power > 0, price below 1d EMA34, volume spike
# Uses Elder Ray to detect momentum shifts and 1d EMA for trend alignment.
# Works in bull/bear markets by following higher timeframe trend with momentum confirmation.
# Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate EMA13 for Elder Ray (on 6h data)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # High - EMA13
    bear_power = low - ema13   # Low - EMA13
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):  # Start from 13 to ensure EMA13 is valid
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0, Bear Power < 0, price above 1d EMA, volume spike
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                close[i] > ema_34_1d_aligned[i] and volume[i] > 1.8 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bull Power < 0, Bear Power > 0, price below 1d EMA, volume spike
            elif (bull_power[i] < 0 and bear_power[i] > 0 and 
                  close[i] < ema_34_1d_aligned[i] and volume[i] > 1.8 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Elder Ray divergence or price crosses 1d EMA
            if position == 1:
                # Exit long: Bear Power becomes positive OR price below 1d EMA
                if bear_power[i] >= 0 or close[i] < ema_34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Bull Power becomes positive OR price above 1d EMA
                if bull_power[i] >= 0 or close[i] > ema_34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dEMA34_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0