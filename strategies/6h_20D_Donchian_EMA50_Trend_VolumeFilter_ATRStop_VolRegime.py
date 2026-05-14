# 6h_20D_Donchian_EMA50_Trend_VolumeFilter_ATRStop_VolRegime
# Hypothesis: This strategy uses 20-day Donchian breakouts aligned with 50-day EMA trend filter,
# volume confirmation, and volatility filtering to capture medium-term trends.
# The 6h timeframe balances responsiveness with noise reduction, and the multi-timeframe
# approach (daily for trend/levels, 6h for execution) reduces false breakouts.
# Volatility regime filter (current 6h ATR < 1.5 * 14d ATR) avoids extreme volatility periods.
# Designed to work in both bull and bear markets by requiring trend alignment and
# only taking breakouts in the direction of the 50-day EMA.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE for 1d indicators
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-day ATR for volatility filter
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-day Donchian channels (breakout levels)
    highest_20d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_20d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 50-day EMA of daily close (long-term trend filter)
    ema_50d_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all 1d indicators to 6h timeframe
    atr_14d_aligned = align_htf_to_ltf(prices, df_1d, atr_14d)
    highest_20d_aligned = align_htf_to_ltf(prices, df_1d, highest_20d)
    lowest_20d_aligned = align_htf_to_ltf(prices, df_1d, lowest_20d)
    ema_50d_aligned = align_htf_to_ltf(prices, df_1d, ema_50d_1d)
    
    # Calculate 6h ATR for volatility filter and stop sizing
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_6h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Precompute hour of day for session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Volume filter: 6h volume > 20-period average
    volume = prices['volume'].values
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(atr_14d_aligned[i]) or np.isnan(highest_20d_aligned[i]) or np.isnan(lowest_20d_aligned[i]) or np.isnan(ema_50d_aligned[i]) or np.isnan(atr_6h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter
        vol_filter = volume[i] > volume_ma_20[i]
        
        # Price levels
        resistance = highest_20d_aligned[i]
        support = lowest_20d_aligned[i]
        trend_filter = ema_50d_aligned[i]
        price = close[i]
        
        # Volatility filter: current 6h ATR < 1.5 * 14d ATR (avoid extreme volatility)
        vol_filter_2 = atr_6h[i] < 1.5 * atr_14d_aligned[i]
        
        if position == 0:
            # Long: price breaks above 20-day resistance, above 50-day EMA, with volume and volatility filter
            if price > resistance and price > trend_filter and vol_filter and vol_filter_2:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below 20-day support, below 50-day EMA, with volume and volatility filter
            elif price < support and price < trend_filter and vol_filter and vol_filter_2:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: stop loss (2x ATR below entry) or price breaks below 20-day support
            if price <= entry_price - 2.0 * atr_6h[i] or price < support:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: stop loss (2x ATR above entry) or price breaks above 20-day resistance
            if price >= entry_price + 2.0 * atr_6h[i] or price > resistance:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_20D_Donchian_EMA50_Trend_VolumeFilter_ATRStop_VolRegime"
timeframe = "6h"
leverage = 1.0