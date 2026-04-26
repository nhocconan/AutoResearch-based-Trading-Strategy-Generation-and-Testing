#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike_ATRStop_v1
Hypothesis: On 4h timeframe, trade Donchian(20) breakouts with 1d EMA34 trend filter and volume spike confirmation. Daily trend filter reduces false breakouts in ranging markets. Volume spike confirms institutional participation. ATR-based stoploss manages risk. Designed for 75-200 total trades over 4 years (19-50/year) with discrete sizing (0.30) to minimize fee drag. Works in bull/bear markets via daily trend filter.
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
    
    # Get 1d data for EMA(34) trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Donchian channels (20-period) from 1d data
    # Donchian upper = highest high of last 20 days
    # Donchian lower = lowest low of last 20 days
    high_20 = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current 1d volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_spike = df_1d['volume'].values > (2.0 * vol_ma)
    
    # ATR(14) for stoploss calculation on 1d
    tr1 = np.abs(df_1d['high'].values - df_1d['low'].values)
    tr2 = np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1))
    tr3 = np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))
    tr1[0] = tr2[0] = tr3[0] = 0  # first period
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align HTF indicators to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of EMA(34) 1d, Donchian(20), volume MA(20), ATR(14)
    start_idx = max(34, 20, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(high_20_aligned[i]) or
            np.isnan(low_20_aligned[i]) or
            np.isnan(volume_spike_aligned[i]) or
            np.isnan(atr_14_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        ema_34_1d_val = ema_34_1d_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol_spike = volume_spike_aligned[i] > 0.5  # convert back to boolean
        upper_channel = high_20_aligned[i]
        lower_channel = low_20_aligned[i]
        atr_val = atr_14_aligned[i]
        
        # Trend filter: price > EMA34 (uptrend) or < EMA34 (downtrend)
        uptrend = close_val > ema_34_1d_val
        downtrend = close_val < ema_34_1d_val
        
        if position == 0:
            # Long: break above upper Donchian channel with uptrend and volume spike
            long_signal = (high_val > upper_channel) and uptrend and vol_spike
            
            # Short: break below lower Donchian channel with downtrend and volume spike
            short_signal = (low_val < lower_channel) and downtrend and vol_spike
            
            if long_signal:
                signals[i] = 0.30
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.30
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.30
            # Exit: trend reversal or ATR-based stoploss
            if close_val < ema_34_1d_val or close_val < entry_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.30
            # Exit: trend reversal or ATR-based stoploss
            if close_val > ema_34_1d_val or close_val > entry_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0