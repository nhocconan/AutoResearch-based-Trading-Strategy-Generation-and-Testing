#!/usr/bin/env python3
"""
4h_donchian_breakout_1d_trend_volume_v4
Hypothesis: On 4h timeframe, use Donchian(20) breakouts for entry signals, filtered by 1d EMA trend and volume confirmation. 
In bull markets, breakout above upper band with volume confirms momentum; in bear markets, breakout below lower band with volume confirms short opportunities. 
Volume confirms genuine breakouts. 1d EMA filter ensures alignment with higher timeframe trend, reducing whipsaw.
Tightened entry conditions: require volume > 2.0x average and ATR(14) > 1.5x ATR(50) to filter low-volatility breakouts.
Target: 20-40 trades/year (~80-160 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v4"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50 = df_1d['close'].ewm(span=50, adjust=False).mean()
    
    # Align 1d EMA50 to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50.values)
    
    # Donchian(20) on 4h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper = high_series.rolling(window=20, min_periods=20).max().values
    lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # ATR(14) and ATR(50) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first value
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Volume confirmation (20-period average on 4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(atr14[i]) or np.isnan(atr50[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR(14) > 1.5 * ATR(50) to avoid low-volatility breakouts
        vol_filter = atr14[i] > 1.5 * atr50[i]
        
        # Volume confirmation: current volume > 2.0x average volume
        vol_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below lower Donchian band or trend changes
            if close[i] < lower[i] or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian band or trend changes
            if close[i] > upper[i] or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above upper Donchian band, with volume and volatility confirmation
            if (close[i] > upper[i] and vol_confirm and vol_filter and 
                close[i] > ema_50_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower Donchian band, with volume and volatility confirmation
            elif (close[i] < lower[i] and vol_confirm and vol_filter and 
                  close[i] < ema_50_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals