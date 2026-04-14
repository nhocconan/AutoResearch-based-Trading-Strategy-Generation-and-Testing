#!/usr/bin/env python3
"""
4H_1D_4W_Donchian_Breakout_Volume_Regime - Strategy based on Donchian breakouts aligned with 
multi-timeframe trend (1D/4W) with volume confirmation and volatility regime filter.
Designed for low trade frequency (<400 total) and robust performance in bull/bear markets.
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
    
    # Load multi-timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_4w = get_htf_data(prices, '4w')
    
    # Calculate Donchian channels (20-period) on 4H
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max()
    donchian_low = low_series.rolling(window=20, min_periods=20).min()
    
    # Calculate 1D EMA trend (50-period)
    ema_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean()
    
    # Calculate 4W EMA trend (20-period)
    ema_4w = pd.Series(df_4w['close']).ewm(span=20, adjust=False, min_periods=20).mean()
    
    # Calculate average volume (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    # Calculate volatility regime using ATR ratio
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean()
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean()
    volatility_regime = atr / atr_ma  # >1 = high volatility, <1 = low volatility
    
    # Align multi-timeframe indicators
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d.values)
    ema_4w_aligned = align_htf_to_ltf(prices, df_4w, ema_4w.values)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after sufficient data
    start = 100
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(ema_1d_aligned[i]) or 
            np.isnan(ema_4w_aligned[i]) or 
            np.isnan(vol_avg[i]) or 
            np.isnan(volatility_regime[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        # Multi-timeframe trend alignment: price above both EMAs = bullish, below both = bearish
        bullish_alignment = price > ema_1d_aligned[i] and price > ema_4w_aligned[i]
        bearish_alignment = price < ema_1d_aligned[i] and price < ema_4w_aligned[i]
        
        if position == 0:
            # Long entry: Donchian breakout above + bullish alignment + volume + volatility expansion
            if (price > donchian_high[i] and 
                bullish_alignment and 
                vol > vol_threshold and 
                volatility_regime[i] > 1.0):
                position = 1
                signals[i] = position_size
            # Short entry: Donchian breakdown below + bearish alignment + volume + volatility expansion
            elif (price < donchian_low[i] and 
                  bearish_alignment and 
                  vol > vol_threshold and 
                  volatility_regime[i] > 1.0):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Donchian breakdown below OR trend reversal
            if price < donchian_low[i] or not bullish_alignment:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Donchian breakout above OR trend reversal
            if price > donchian_high[i] or not bearish_alignment:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4H_1D_4W_Donchian_Breakout_Volume_Regime"
timeframe = "4h"
leverage = 1.0