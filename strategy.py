#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA13 trend filter and volume confirmation
# Elder Ray measures bull/bear power relative to EMA13 to identify trend strength.
# 1d EMA13 filter ensures alignment with daily trend for higher probability trades.
# Volume spike (>1.5x 20-period average) confirms institutional participation.
# Designed for 6h timeframe targeting 15-30 trades/year with strong performance in both bull and bear markets.
# Works in bull markets via bull power strength and in bear markets via bear power reversals.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA13 trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA(13) for trend filter
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Calculate 13-period EMA for Elder Ray (using close prices)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(ema_13_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Strong bull power + daily uptrend + volume confirmation
            if (bull_power[i] > 0 and  # Bullish momentum
                close[i] > ema_13_1d_aligned[i] and  # Daily uptrend
                volume[i] > 1.5 * vol_avg_20[i]):    # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Strong bear power + daily downtrend + volume confirmation
            elif (bear_power[i] < 0 and   # Bearish momentum
                  close[i] < ema_13_1d_aligned[i] and  # Daily downtrend
                  volume[i] > 1.5 * vol_avg_20[i]):    # Volume confirmation
                signals[i] = -0.25
                position = -1
        else:
            # Exit: momentum divergence or trend reversal
            if position == 1:
                # Exit long: bear power turns positive (momentum fading) or trend turns down
                if (bear_power[i] >= 0 or 
                    close[i] < ema_13_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: bull power turns negative (momentum fading) or trend turns up
                if (bull_power[i] <= 0 or 
                    close[i] > ema_13_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dEMA13_VolumeConfirm"
timeframe = "6h"
leverage = 1.0