# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
4h_donchian_breakout_1w_trend_volume_v1
Hypothesis: Breakouts of 20-period Donchian channels on 4h, filtered by 1-week EMA200 trend and volume confirmation, capture trending moves in both bull and bear markets. Volatility filter avoids choppy periods. Target: 20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1w_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA200 for trend filter
    ema_200 = df_1w['close'].ewm(span=200, adjust=False).mean()
    
    # Align 1w EMA200 to 4h timeframe
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200.values)
    
    # Donchian Channel (20-period) on 4h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper = high_series.rolling(window=20, min_periods=20).max()
    lower = low_series.rolling(window=20, min_periods=20).min()
    
    # Donchian breakout signals
    breakout_up = (close > upper.shift(1)).astype(float)  # Break above prior upper band
    breakout_dn = (close < lower.shift(1)).astype(float)  # Break below prior lower band
    
    # Volatility filter: ATR(14) < 50th percentile of ATR(50) to avoid chop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    atr_median = pd.Series(atr).rolling(window=50, min_periods=50).median()
    vol_filter = (atr < atr_median).astype(float)  # Low volatility regime
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = (volume > 1.5 * vol_ma).astype(float)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema_200_aligned[i]) or np.isnan(vol_ma[i]) or 
            vol_ma[i] <= 0 or np.isnan(atr[i]) or np.isnan(atr_median[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below lower Donchian band or trend fails
            if close[i] < lower.iloc[i] or close[i] < ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian band or trend fails
            if close[i] > upper.iloc[i] or close[i] > ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Donchian breakout up, with trend, volume, and low volatility
            if (breakout_up.iloc[i] and 
                close[i] > ema_200_aligned[i] and 
                vol_confirm.iloc[i] and 
                vol_filter.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Donchian breakout down, with trend, volume, and low volatility
            elif (breakout_dn.iloc[i] and 
                  close[i] < ema_200_aligned[i] and 
                  vol_confirm.iloc[i] and 
                  vol_filter.iloc[i]):
                position = -1
                signals[i] = -0.25
    
    return signals