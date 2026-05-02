#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Bull/Bear Power with 1d EMA34 trend filter and volume spike confirmation
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# In strong uptrends: Bull Power > 0 and increasing, Bear Power < 0
# In strong downtrends: Bear Power < 0 and decreasing, Bull Power > 0
# 1d EMA34 filter ensures trades align with higher-timeframe trend to avoid counter-trend whipsaws
# Volume spike (>1.8 x 20-period EMA) confirms momentum validity
# Discrete position sizing (0.25) balances opportunity with fee drag control
# Target: 50-150 total trades over 4 years (12-37/year) for optimal risk-adjusted returns

name = "6h_ElderRay_BullBearPower_1dEMA34_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation (volume spike > 1.8 x 20-period EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (1.8 * vol_ema_20)
    
    # 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Elder Ray calculations (requires EMA13 on 6h data)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for EMA13 and EMA34)
    start_idx = 40
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Strong bull power (bullish momentum) with volume confirmation and uptrend
            if bull_power[i] > 0 and bull_power[i] > bear_power[i] * -1 and volume_confirmation[i] and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Strong bear power (bearish momentum) with volume confirmation and downtrend
            elif bear_power[i] < 0 and abs(bear_power[i]) > bull_power[i] and volume_confirmation[i] and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bear power becomes stronger than bull power (momentum shift) OR trend changes to downtrend
            if abs(bear_power[i]) > bull_power[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bull power becomes stronger than bear power (momentum shift) OR trend changes to uptrend
            if bull_power[i] > abs(bear_power[i]) or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals