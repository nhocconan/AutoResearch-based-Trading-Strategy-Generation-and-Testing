#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d EMA34 trend filter + volume confirmation.
# Long when Alligator jaws < teeth < lips (bullish alignment) AND price > 1d EMA34 AND volume > 2.0x 20-bar average.
# Short when Alligator jaws > teeth > lips (bearish alignment) AND price < 1d EMA34 AND volume > 2.0x 20-bar average.
# Uses discrete position sizing (0.25) to limit drawdown and fee churn. Works in bull/bear via 1d EMA34 trend filter.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_WilliamsAlligator_1dEMA34_Trend_VolumeConfirm_v1"
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
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator on 6h: SMMA(13,8), SMMA(8,5), SMMA(5,3) of median price
    median_price = (high + low) / 2.0
    jaws = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Volume confirmation: volume > 2.0x 20-period average (stricter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 21)  # warmup for Alligator (13+8)
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(ema_34_1d_aligned[i]):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            continue
        
        is_bullish_alligator = jaws[i] < teeth[i] and teeth[i] < lips[i]
        is_bearish_alligator = jaws[i] > teeth[i] and teeth[i] > lips[i]
        is_uptrend = close[i] > ema_34_1d_aligned[i]
        is_downtrend = close[i] < ema_34_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: bullish Alligator + uptrend + volume confirmation
            if is_bullish_alligator and is_uptrend and curr_volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator + downtrend + volume confirmation
            elif is_bearish_alligator and is_downtrend and curr_volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit when bullish alignment breaks OR trend reverses
            if not is_bullish_alligator or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when bearish alignment breaks OR trend reverses
            if not is_bearish_alligator or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals