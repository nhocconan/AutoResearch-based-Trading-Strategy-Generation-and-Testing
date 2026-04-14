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
    
    # Weekly EMA(34) for long-term trend
    close_1w_series = pd.Series(close_1w)
    ema_34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily ATR for volatility filter (14-period)
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    high_close[0] = high_low[0]
    low_close[0] = high_low[0]
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr_series = pd.Series(tr)
    atr = tr_series.rolling(window=14, min_periods=14).mean().values
    
    # Daily volume filter: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    # Daily Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(60, n):
        # Skip if any critical data is NaN
        if np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr[i]) or np.isnan(donchian_high[i]) or \
           np.isnan(donchian_low[i]) or np.isnan(vol_ma[i]):
            continue
        
        # Long conditions: Price above weekly EMA + Donchian breakout + volume
        if position == 0:
            if (close[i] > ema_34_1w_aligned[i] and
                close[i] > donchian_high[i] and
                volume[i] > vol_ma[i] * 1.5):
                position = 1
                signals[i] = position_size
        
        # Short conditions: Price below weekly EMA + Donchian breakdown + volume
        elif position == 0:
            if (close[i] < ema_34_1w_aligned[i] and
                close[i] < donchian_low[i] and
                volume[i] > vol_ma[i] * 1.5):
                position = -1
                signals[i] = -position_size
        
        # Exit long: Price crosses below weekly EMA or Donchian low
        elif position == 1:
            if close[i] < ema_34_1w_aligned[i] or close[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
        
        # Exit short: Price crosses above weekly EMA or Donchian high
        elif position == -1:
            if close[i] > ema_34_1w_aligned[i] or close[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "1d_weekly_EMA34_Donchian_Volume"
timeframe = "1d"
leverage = 1.0