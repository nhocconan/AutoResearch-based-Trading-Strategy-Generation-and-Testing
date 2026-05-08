#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1-day trend filter and volume spike.
# Bull Power = High - EMA13, Bear Power = Low - EMA13. Long when Bull Power > 0 and rising, Bear Power < 0 and falling.
# Short when Bear Power < 0 and falling, Bull Power < 0 and rising.
# Uses 1-day EMA34 for trend filter and volume spike for confirmation.
# Designed to capture institutional buying/selling pressure in both bull and bear markets.
# Target: 15-30 trades/year to minimize fee drag.

name = "6h_ElderRay_Trend_1dEMA34_Volume"
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
    
    # 1-day trend: EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    # Using 13-period EMA on 6h close for consistency with classic Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 and rising (bullish momentum), Bear Power < 0, above 1d EMA34, volume spike
            bull_rising = bull_power[i] > bull_power[i-1]
            long_cond = (bull_power[i] > 0) and bull_rising and \
                        (bear_power[i] < 0) and \
                        (close[i] > ema_34_1d_aligned[i]) and \
                        volume_spike[i]
            
            # Short: Bear Power < 0 and falling (bearish momentum), Bull Power < 0, below 1d EMA34, volume spike
            bear_falling = bear_power[i] < bear_power[i-1]
            short_cond = (bear_power[i] < 0) and bear_falling and \
                         (bull_power[i] < 0) and \
                         (close[i] < ema_34_1d_aligned[i]) and \
                         volume_spike[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bear Power turns positive (bulls losing control) or price below 1d EMA34
            if (bear_power[i] > 0) or (close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bull Power turns positive (bears losing control) or price above 1d EMA34
            if (bull_power[i] > 0) or (close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals