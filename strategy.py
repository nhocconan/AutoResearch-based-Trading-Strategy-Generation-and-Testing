#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Donchian channels (20-period)
    high_series = pd.Series(df_1d['high'])
    low_series = pd.Series(df_1d['low'])
    upper_donch = high_series.rolling(window=20, min_periods=20).max().values
    lower_donch = low_series.rolling(window=20, min_periods=20).min().values
    upper_donch_aligned = align_htf_to_ltf(prices, df_1d, upper_donch)
    lower_donch_aligned = align_htf_to_ltf(prices, df_1d, lower_donch)
    
    # Calculate 1d ATR(14) for volatility filter
    high_series_1d = pd.Series(df_1d['high'])
    low_series_1d = pd.Series(df_1d['low'])
    close_series_1d = pd.Series(df_1d['close'])
    tr1 = high_series_1d - low_series_1d
    tr2 = abs(high_series_1d - close_series_1d.shift(1))
    tr3 = abs(low_series_1d - close_series_1d.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 1d ADX(14) for trend strength filter
    plus_dm = high_series_1d.diff()
    minus_dm = low_series_1d.diff().multiply(-1)
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr_atr = tr.rolling(window=14, min_periods=14).mean()
    plus_di = 100 * (plus_dm.rolling(window=14, min_periods=14).mean() / tr_atr)
    minus_di = 100 * (minus_dm.rolling(window=14, min_periods=14).mean() / tr_atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 200
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_donch_aligned[i]) or 
            np.isnan(lower_donch_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Volatility filter: avoid extremely low volatility periods
        vol_filter = atr_1d_aligned[i] > (price * 0.01)  # Minimum 1% ATR relative to price
        
        # Trend strength filter: only trade when ADX > 25
        trend_filter = adx_aligned[i] > 25
        
        if position == 0:
            # Long setup: price breaks above 1d upper Donchian + volatility + trend strength
            if (price > upper_donch_aligned[i] and vol_filter and trend_filter):
                position = 1
                signals[i] = position_size
            # Short setup: price breaks below 1d lower Donchian + volatility + trend strength
            elif (price < lower_donch_aligned[i] and vol_filter and trend_filter):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below 1d lower Donchian (reversal signal)
            if price < lower_donch_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above 1d upper Donchian (reversal signal)
            if price > upper_donch_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1dDonchian20_Volatility_ADX_Filter"
timeframe = "4h"
leverage = 1.0