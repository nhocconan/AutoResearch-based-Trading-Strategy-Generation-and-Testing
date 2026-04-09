#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index + 1w trend filter + volume confirmation
# - Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
# - Long when Bull Power > 0 AND Bear Power rising (bullish momentum)
# - Short when Bear Power < 0 AND Bull Power falling (bearish momentum)
# - 1w EMA34 trend filter: only long in weekly uptrend, short in weekly downtrend
# - Volume confirmation: 6h volume > 20-period median volume
# - Works in bull/bear: Elder Ray captures momentum, weekly filter avoids counter-trend trades
# - Target: 50-150 total trades over 4 years (12-37/year) per 6h strategy guidelines

name = "6h_1w_elderray_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 35 or len(df_1d) < 35:
        return np.zeros(n)
    
    # Pre-compute 1w indicators
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    weekly_uptrend = ema_34_1w > np.roll(ema_34_1w, 1)  # weekly EMA rising
    weekly_downtrend = ema_34_1w < np.roll(ema_34_1w, 1)  # weekly EMA falling
    
    # Pre-compute 1d indicators for volume regime
    volume_1d = df_1d['volume'].values
    median_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).median().values
    volume_regime_1d = volume_1d > median_volume_20  # high volume days
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 6h EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # High - EMA13
    bear_power = ema_13 - low   # EMA13 - Low
    
    # Elder Ray momentum: rising/falling power
    bull_power_rising = bull_power > np.roll(bull_power, 1)
    bull_power_falling = bull_power < np.roll(bull_power, 1)
    bear_power_rising = bear_power > np.roll(bear_power, 1)
    bear_power_falling = bear_power < np.roll(bear_power, 1)
    
    # Align all HTF indicators to 6h timeframe (completed bars only)
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend)
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend)
    volume_regime_aligned = align_htf_to_ltf(prices, df_1d, volume_regime_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(weekly_uptrend_aligned[i]) or
            np.isnan(weekly_downtrend_aligned[i]) or
            np.isnan(volume_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Bull Power turns negative OR Bear Power rises sharply
            if bull_power[i] <= 0 or bear_power_falling[i] == False and bear_power[i] > 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bear Power turns positive OR Bull Power rises sharply
            if bear_power[i] >= 0 or bull_power_rising[i] == False and bull_power[i] < 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Elder Ray signals with volume and weekly trend confirmation
            # Long: Bull Power > 0 AND Bear Power rising AND weekly uptrend AND volume regime
            if (bull_power[i] > 0 and 
                bear_power_rising[i] and 
                weekly_uptrend_aligned[i] and 
                volume_regime_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: Bear Power < 0 AND Bull Power falling AND weekly downtrend AND volume regime
            elif (bear_power[i] < 0 and 
                  bull_power_falling[i] and 
                  weekly_downtrend_aligned[i] and 
                  volume_regime_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals