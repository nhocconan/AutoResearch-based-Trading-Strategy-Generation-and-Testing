#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator with 1d EMA trend filter and volume confirmation.
# Long when price > Alligator Jaw AND Alligator Teeth > Alligator Lips (bullish alignment) AND 1d EMA50 > EMA200 AND 1d volume > 1.5 * 20-period average volume.
# Short when price < Alligator Jaw AND Alligator Teeth < Alligator Lips (bearish alignment) AND 1d EMA50 < EMA200 AND 1d volume > 1.5 * 20-period average volume.
# Exit when price crosses back over Alligator Jaw.
# Uses discrete position sizing (0.30) to limit fee churn. Designed for 12h timeframe with strict entry conditions.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h.

name = "12h_WilliamsAlligator_1dEMA50_Trend_1dVolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 and EMA200 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_trend = align_htf_to_ltf(prices, df_1d, ema_50 > ema_200)  # Boolean: True for bullish, False for bearish
    
    # Calculate 1d volume confirmation filter (HTF)
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = volume_1d > (1.5 * vol_ma_20_1d)
    volume_confirm_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm_1d.astype(float))
    
    # Williams Alligator: SMAs of median price (typical price) with specific periods
    typical_price = (high + low + close) / 3.0
    jaw = pd.Series(typical_price).rolling(window=13, min_periods=13).mean().shift(8).values  # 13-period, shifted 8
    teeth = pd.Series(typical_price).rolling(window=8, min_periods=8).mean().shift(5).values    # 8-period, shifted 5
    lips = pd.Series(typical_price).rolling(window=5, min_periods=5).mean().shift(3).values    # 5-period, shifted 3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_trend[i]) or np.isnan(volume_confirm_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price > jaw AND teeth > lips (bullish alignment) AND bullish trend AND volume confirmation
            if (close[i] > jaw[i] and teeth[i] > lips[i] and 
                ema_trend[i] and 
                volume_confirm_1d_aligned[i] > 0.5):
                signals[i] = 0.30
                position = 1
            # SHORT: price < jaw AND teeth < lips (bearish alignment) AND bearish trend AND volume confirmation
            elif (close[i] < jaw[i] and teeth[i] < lips[i] and 
                  not ema_trend[i] and 
                  volume_confirm_1d_aligned[i] > 0.5):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses back below jaw
            if close[i] < jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: price crosses back above jaw
            if close[i] > jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals