#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1d EMA34 trend filter and volume confirmation
# Elder Ray measures bull/bear power relative to EMA13, capturing institutional buying/selling pressure.
# 1d EMA34 filter ensures alignment with daily trend for higher probability trades.
# Volume confirmation (>1.3x 20-period average) filters weak breakouts.
# Designed for 6h timeframe targeting 15-30 trades/year, effective in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA34 trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Elder Ray components on 6h data
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull power positive + price above 1d EMA34 + volume confirmation
            if (bull_power[i] > 0 and
                close[i] > ema_34_1d_aligned[i] and
                volume[i] > 1.3 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear power negative + price below 1d EMA34 + volume confirmation
            elif (bear_power[i] < 0 and
                  close[i] < ema_34_1d_aligned[i] and
                  volume[i] > 1.3 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Elder power divergence or trend reversal
            if position == 1:
                # Exit long: bear power turns negative or trend turns down
                if (bear_power[i] < 0 or
                    close[i] < ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: bull power turns positive or trend turns up
                if (bull_power[i] > 0 or
                    close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dEMA34_VolumeConfirm"
timeframe = "6h"
leverage = 1.0