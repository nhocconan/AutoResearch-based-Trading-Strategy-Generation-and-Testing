#!/usr/bin/env python3
"""
4h Williams Alligator + 1d EMA34 Trend + Volume Spike + ATR Filter
Hypothesis: Williams Alligator identifies trend phases on 4h. Daily EMA34 filters for higher timeframe trend alignment.
Volume spike confirms breakout strength. ATR-based stoploss limits drawdown during choppy periods.
Works in bull/bear by following daily trend while using Alligator for entry timing.
Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator on 4h: SMAs of median price
    median_price = (high + low) / 2.0
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values  # 13-period, shifted 8
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values   # 8-period, shifted 5
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values    # 5-period, shifted 3
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 4h ATR for volatility filter and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 4h volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Alligator (13+8=21) + EMA34 + ATR14 + VolMA20
    start_idx = max(21, 34, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_34_level = ema_34_1d_aligned[i]
        atr_value = atr[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        volume_spike = curr_volume > 2.0 * vol_ma_20[i]
        
        # Alligator conditions: lips > teeth > jaw = bullish alignment
        # lips < teeth < jaw = bearish alignment
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Update signals for existing positions
        if position == 1:
            signals[i] = 0.25
            # ATR-based stoploss: exit if price drops below entry - 2*ATR
            if curr_close < entry_price - 2.0 * atr_value:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        elif position == -1:
            signals[i] = -0.25
            # ATR-based stoploss: exit if price rises above entry + 2*ATR
            if curr_close > entry_price + 2.0 * atr_value:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        # Entry conditions: Alligator alignment + trend + volume
        if position == 0:
            long_condition = bullish_alignment and (curr_close > ema_34_level) and volume_spike
            short_condition = bearish_alignment and (curr_close < ema_34_level) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
    
    return signals

name = "4h_Williams_Alligator_1dEMA34_Trend_VolumeSpike_ATR_v1"
timeframe = "4h"
leverage = 1.0