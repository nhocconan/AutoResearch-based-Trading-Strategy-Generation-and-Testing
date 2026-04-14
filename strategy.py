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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Donchian channels (20-period high/low)
    high_series = pd.Series(df_1d['high'])
    low_series = pd.Series(df_1d['low'])
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate 1d ATR(14) for volatility filter
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1)) if 'close_series' in locals() else abs(high_series - pd.Series(close).shift(1))
    tr3 = abs(low_series - close_series.shift(1)) if 'close_series' in locals() else abs(low_series - pd.Series(close).shift(1))
    if 'close_series' not in locals():
        close_series = pd.Series(close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 1d ADX(14) for trend strength
    plus_dm = high_series.diff()
    minus_dm = low_series.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    tr_14 = tr.rolling(window=14, min_periods=14).sum()
    plus_di_14 = 100 * (plus_dm.rolling(window=14, min_periods=14).sum() / tr_14)
    minus_di_14 = 100 * (minus_dm.rolling(window=14, min_periods=14).sum() / tr_14)
    dx = (abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)) * 100
    adx_1d = dx.rolling(window=14, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 100
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Volatility filter: ATR > 0.5% of price
        vol_filter = (atr_1d_aligned[i] / price) > 0.005 if price > 0 else False
        
        # Trend strength filter: ADX > 25
        trend_filter = adx_1d_aligned[i] > 25
        
        if position == 0:
            # Long breakout: price breaks above 1d Donchian high
            if price > donchian_high_aligned[i] and vol_filter and trend_filter:
                position = 1
                signals[i] = position_size
            # Short breakdown: price breaks below 1d Donchian low
            elif price < donchian_low_aligned[i] and vol_filter and trend_filter:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below 1d Donchian low
            if price < donchian_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above 1d Donchian high
            if price > donchian_high_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1dDonchian20_ADX25_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0