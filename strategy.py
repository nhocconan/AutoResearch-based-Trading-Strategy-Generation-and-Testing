#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d EMA34 trend filter and 1w volume regime filter.
# Long when Bull Power > 0 AND Bear Power < 0 (bullish divergence) AND price > 1d EMA34 AND 1w volume > 1.2 * 20-period average volume.
# Short when Bear Power < 0 AND Bull Power > 0 (bearish divergence) AND price < 1d EMA34 AND 1w volume > 1.2 * 20-period average volume.
# Exit when Bull Power <= 0 (for longs) or Bear Power >= 0 (for shorts).
# Uses discrete position sizing (0.25) to limit fee churn. Designed for BTC/ETH robustness by measuring bull/bear power relative to EMA13 with trend and volume confirmation.
# Target: 60-100 total trades over 4 years (15-25/year) for 6h timeframe.

name = "6h_ElderRay_TrendFilter_1wVolumeRegime_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1w volume regime filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    volume_1w = df_1w['volume'].values
    vol_ma_20 = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_regime = volume_1w > (1.2 * vol_ma_20)
    volume_regime_aligned = align_htf_to_ltf(prices, df_1w, volume_regime.astype(float))
    
    # Calculate Elder Ray Index (13-period EMA) on primary timeframe
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):  # Start after EMA13 warmup
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_regime_aligned[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power > 0 AND Bear Power < 0 (bullish divergence) AND price > 1d EMA34 AND high volume regime
            if (bull_power[i] > 0 and 
                bear_power[i] < 0 and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_regime_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power < 0 AND Bull Power > 0 (bearish divergence) AND price < 1d EMA34 AND high volume regime
            elif (bear_power[i] < 0 and 
                  bull_power[i] > 0 and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_regime_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull Power <= 0 (loss of bullish momentum)
            if bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power >= 0 (loss of bearish momentum)
            if bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals