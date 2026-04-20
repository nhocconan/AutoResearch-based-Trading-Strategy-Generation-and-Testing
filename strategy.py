#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA trend filter and volume confirmation
# In bull markets: buy breakouts above 20-period high when above 12h EMA (uptrend)
# In bear markets: sell breakdowns below 20-period low when below 12h EMA (downtrend)
# Volume filter ensures breakouts have participation
# Uses discrete position sizing (0.25) to minimize fee churn
# Target: 100-200 total trades over 4 years (25-50/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(34) for trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Load daily data ONCE for Donchian channels (using 20-day period)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-day Donchian channels
    highest_20d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_20d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align daily Donchian levels to 4h timeframe
    highest_20d_aligned = align_htf_to_ltf(prices, df_1d, highest_20d)
    lowest_20d_aligned = align_htf_to_ltf(prices, df_1d, lowest_20d)
    
    # Calculate 4h ATR for volatility filter and stop sizing
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_4h = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Precompute hour of day for session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Volume filter: 4h volume > 20-period average
    volume = prices['volume'].values
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(highest_20d_aligned[i]) or np.isnan(lowest_20d_aligned[i]) or \
           np.isnan(ema_34_12h_aligned[i]) or np.isnan(atr_4h[i]):
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
        ema_trend = ema_34_12h_aligned[i]
        price = close[i]
        
        if position == 0:
            # Long: price breaks above 20-day resistance, above 12h EMA (uptrend), with volume
            if price > resistance and price > ema_trend and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below 20-day support, below 12h EMA (downtrend), with volume
            elif price < support and price < ema_trend and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: stop loss (2x ATR below entry) or price breaks below 20-day support
            if price <= entry_price - 2.0 * atr_4h[i] or price < support:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: stop loss (2x ATR above entry) or price breaks above 20-day resistance
            if price >= entry_price + 2.0 * atr_4h[i] or price > resistance:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_20D_Donchian_12hEMA34_VolumeFilter"
timeframe = "4h"
leverage = 1.0