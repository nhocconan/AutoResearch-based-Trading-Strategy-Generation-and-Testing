#!/usr/bin/env python3
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
    
    # Load weekly data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # 60-week EMA for long-term trend
    close_1w_series = pd.Series(close_1w)
    ema_60w = close_1w_series.ewm(span=60, adjust=False, min_periods=60).mean().values
    
    # 20-week ATR for volatility measurement
    high_low = high_1w - low_1w
    high_close = np.abs(high_1w - np.roll(close_1w, 1))
    low_close = np.abs(low_1w - np.roll(close_1w, 1))
    high_close[0] = high_low[0]
    low_close[0] = high_low[0]
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr_series = pd.Series(tr)
    atr_20w = tr_series.rolling(window=20, min_periods=20).mean().values
    
    # Align weekly indicators to 6h timeframe
    ema_60w_6h = align_htf_to_ltf(prices, df_1w, ema_60w)
    atr_20w_6h = align_htf_to_ltf(prices, df_1w, atr_20w)
    
    # 6h ATR for position sizing and volatility filter
    high_low_6h = high - low
    high_close_6h = np.abs(high - np.roll(close, 1))
    low_close_6h = np.abs(low - np.roll(close, 1))
    high_close_6h[0] = high_low_6h[0]
    low_close_6h[0] = high_low_6h[0]
    tr_6h = np.maximum(high_low_6h, np.maximum(high_close_6h, low_close_6h))
    tr_6h_series = pd.Series(tr_6h)
    atr_14 = tr_6h_series.rolling(window=14, min_periods=14).mean().values
    
    # 6h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_60w_6h[i]) or np.isnan(atr_20w_6h[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr_14[i])):
            continue
        
        # Trend filter: price above/below 60-week EMA
        uptrend = close[i] > ema_60w_6h[i]
        downtrend = close[i] < ema_60w_6h[i]
        
        # Volatility filter: current ATR > 1.5x 20-week ATR (expansion)
        vol_expansion = atr_14[i] > (atr_20w_6h[i] * 1.5)
        
        if position == 0:
            # Long: Uptrend + volatility expansion + break above Donchian high
            if (uptrend and vol_expansion and 
                close[i] > donchian_high[i] and close[i-1] <= donchian_high[i]):
                position = 1
                signals[i] = position_size
            # Short: Downtrend + volatility expansion + break below Donchian low
            elif (downtrend and vol_expansion and 
                  close[i] < donchian_low[i] and close[i-1] >= donchian_low[i]):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Trend reversal or volatility contraction
            if (not uptrend) or (not vol_expansion) or (close[i] < donchian_low[i]):
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Trend reversal or volatility contraction
            if (not downtrend) or (not vol_expansion) or (close[i] > donchian_high[i]):
                position = 0
                signals[i] = 0.0
    
    return signals

name = "6h_1w_EMA60_ATR20_Donchian20_VolExpansion"
timeframe = "6h"
leverage = 1.0