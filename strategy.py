#!/usr/bin/env python3
"""
1h_Keltner_Channel_Breakout_4hTrend_1dVolFilter
Hypothesis: Price breaking above Keltner upper band in 4h uptrend (price > 4h EMA50) with 1d volume spike triggers long; breaking below lower band in 4h downtrend (price < 4h EMA50) with volume spike triggers short. Uses 1h for entry timing with 4h trend filter and 1d volume confirmation. Designed to capture trend moves with controlled frequency.
"""

name = "1h_Keltner_Channel_Breakout_4hTrend_1dVolFilter"
timeframe = "1h"
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
    
    # Get 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for volume filter (20-period average)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    vol_ma_20_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate Keltner Channel on 1h data (20-period EMA, ATR(10)*2)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    kc_upper = ema_20 + 2 * atr
    kc_lower = ema_20 - 2 * atr
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Apply session filter: only trade between 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            signals[i] = 0.0
            continue
            
        # LONG: Price breaks above Keltner upper band in 4h uptrend with volume spike
        if (ema_50_4h_aligned[i] > 0 and not np.isnan(ema_50_4h_aligned[i]) and
            kc_upper[i] > 0 and not np.isnan(kc_upper[i]) and
            close[i] > kc_upper[i] and
            close[i] > ema_50_4h_aligned[i] and
            vol_ma_20_1d_aligned[i] > 0 and not np.isnan(vol_ma_20_1d_aligned[i]) and
            volume[i] > 2.0 * vol_ma_20_1d_aligned[i]):
            if position != 1:
                signals[i] = 0.20
                position = 1
            else:
                signals[i] = 0.20
        # SHORT: Price breaks below Keltner lower band in 4h downtrend with volume spike
        elif (ema_50_4h_aligned[i] > 0 and not np.isnan(ema_50_4h_aligned[i]) and
              kc_lower[i] > 0 and not np.isnan(kc_lower[i]) and
              close[i] < kc_lower[i] and
              close[i] < ema_50_4h_aligned[i] and
              vol_ma_20_1d_aligned[i] > 0 and not np.isnan(vol_ma_20_1d_aligned[i]) and
              volume[i] > 2.0 * vol_ma_20_1d_aligned[i]):
            if position != -1:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = -0.20
        # EXIT: Price crosses back through EMA20 (middle of Keltner Channel)
        elif position == 1 and close[i] < ema_20[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > ema_20[i]:
            signals[i] = 0.0
            position = 0
        else:
            signals[i] = 0.0
    
    return signals