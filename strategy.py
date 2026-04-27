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
    
    # Get 1d data for trend filter (EMA34) and volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close with proper min_periods
    close_1d = df_1d['close'].values
    ema_34 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        # Use pandas EMA for accuracy and proper min_periods handling
        ema_series = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean()
        ema_34 = ema_series.values
    
    # Align EMA34 to daily timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate ATR(14) for volatility filter with proper min_periods
    tr = np.maximum(high[1:] - low[1:], 
                    np.maximum(np.abs(high[1:] - close[:-1]), 
                               np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.nan], tr])
    atr = np.full(n, np.nan)
    if n >= 15:
        # Use pandas rolling for proper min_periods
        tr_series = pd.Series(tr)
        atr_series = tr_series.rolling(window=14, min_periods=14).mean()
        atr = atr_series.values
    
    # Calculate 20-period volume average with proper min_periods
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    if n >= vol_period:
        volume_series = pd.Series(volume)
        vol_ma_series = volume_series.rolling(window=vol_period, min_periods=vol_period).mean()
        vol_ma = vol_ma_series.values
    
    # Calculate 20-period high/low for Donchian breakout with proper min_periods
    high_max = np.full(n, np.nan)
    low_min = np.full(n, np.nan)
    period = 20
    if n >= period:
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        high_max = high_series.rolling(window=period, min_periods=period).max().values
        low_min = low_series.rolling(window=period, min_periods=period).min().values
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period - ensure all indicators are valid
    start_idx = max(14, vol_period, period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(high_max[i]) or np.isnan(low_min[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        if position == 0:
            # Long: Price breaks above Donchian high with volume AND above 1d EMA34
            if price > high_max[i] and vol_ratio > 2.0 and price > ema_34_aligned[i]:
                signals[i] = size
                position = 1
            # Short: Price breaks below Donchian low with volume AND below 1d EMA34
            elif price < low_min[i] and vol_ratio > 2.0 and price < ema_34_aligned[i]:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below Donchian low or 2x ATR trailing stop
            if price < low_min[i] or price < high_max[i] - 2 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above Donchian high or 2x ATR trailing stop
            if price > high_max[i] or price > low_min[i] + 2 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Donchian20_1dEMA34_Volume_Trend"
timeframe = "1d"
leverage = 1.0