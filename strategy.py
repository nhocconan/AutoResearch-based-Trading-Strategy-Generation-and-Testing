# Hypothesis: 12h Donchian(20) breakout with volume confirmation and 12h EMA(34) trend filter, using 1w/1d HTF for bias. Designed for 12h timeframe to capture medium-term trends in both bull and bear markets while minimizing trade frequency to avoid fee drag. Uses volatility-adjusted position sizing and ATR-based stop loss.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data once for multi-timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-day Donchian channels from daily data (structure)
    highest_20d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_20d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    highest_20d_12h = align_htf_to_ltf(prices, df_1d, highest_20d)
    lowest_20d_12h = align_htf_to_ltf(prices, df_1d, lowest_20d)
    
    # Calculate 34-period EMA on 12h data for trend filter
    close_12h = prices['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 12h ATR for volatility filter and stop sizing
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: 12h volume > 20-period average
    volume_12h = prices['volume'].values
    volume_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # 1-week trend bias: price above/below 50-week EMA
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if NaN in indicators
        if (np.isnan(highest_20d_12h[i]) or np.isnan(lowest_20d_12h[i]) or 
            np.isnan(ema_34_12h[i]) or np.isnan(atr_12h[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_12h[i]
        resistance = highest_20d_12h[i]
        support = lowest_20d_12h[i]
        trend_12h = ema_34_12h[i]
        trend_1w = ema_50_1w_aligned[i]
        vol_filter = volume_12h[i] > volume_ma_20[i]
        
        if position == 0:
            # Long: price breaks above 20-day resistance, above both EMAs, with volume
            if price > resistance and price > trend_12h and price > trend_1w and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below 20-day support, below both EMAs, with volume
            elif price < support and price < trend_12h and price < trend_1w and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: stop loss (2.5x ATR below entry) or price breaks below 20-day support
            if price <= entry_price - 2.5 * atr_12h[i] or price < support:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: stop loss (2.5x ATR above entry) or price breaks above 20-day resistance
            if price >= entry_price + 2.5 * atr_12h[i] or price > resistance:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_20D_Donchian_EMA34_1WTrend_VolumeFilter_ATRStop"
timeframe = "12h"
leverage = 1.0