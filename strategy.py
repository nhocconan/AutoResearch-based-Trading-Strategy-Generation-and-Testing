#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 34-period EMA for trend filter (1d)
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34 = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Average True Range for volatility regime
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(abs(high_1d - pd.Series(close_1d).shift(1)))
    tr3 = pd.Series(abs(low_1d - pd.Series(close_1d).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr14 = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # Volume filter: volume > 1.5x 24-period average (4h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # Donchian channels (20-period) on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34[i]) or np.isnan(atr14[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when volatility is elevated (ATR > 1.5x 50-period MA)
        atr_ma = pd.Series(atr14).rolling(window=50, min_periods=50).mean().values
        high_vol_regime = atr14[i] > (atr_ma[i] * 1.5) if not np.isnan(atr_ma[i]) else False
        
        # Long conditions: price breaks above upper Donchian + trend up + volume spike + high vol regime
        long_breakout = (close[i] > highest_high[i-1] and close[i] > ema34[i] and 
                         volume_spike[i] and high_vol_regime)
        # Short conditions: price breaks below lower Donchian + trend down + volume spike + high vol regime
        short_breakout = (close[i] < lowest_low[i-1] and close[i] < ema34[i] and 
                          volume_spike[i] and high_vol_regime)
        
        if long_breakout:
            signals[i] = 0.25
            position = 1
        elif short_breakout:
            signals[i] = -0.25
            position = -1
        # Exit conditions: opposite Donchian breakout with volume
        elif position == 1 and close[i] < lowest_low[i-1] and volume_spike[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > highest_high[i-1] and volume_spike[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_EMA34_ATRRegime"
timeframe = "4h"
leverage = 1.0