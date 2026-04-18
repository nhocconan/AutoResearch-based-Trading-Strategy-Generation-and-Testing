#!/usr/bin/env python3
"""
1h_Bollinger_Bands_Mean_Reversion_with_Volume_and_Trend_Filter_v1
Hypothesis: Use Bollinger Bands mean reversion on 1h with volume confirmation and 4h/1d trend filter to capture reversals in both bull and bear markets. Bollinger Bands identify overextended moves, volume confirms institutional participation, and higher timeframe filters (4h EMA50, 1d ADX) ensure alignment with stronger trends. Designed for low trade frequency (15-35/year) to minimize fee drift while maintaining edge in ranging and trending markets.
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
    
    # Bollinger Bands (20, 2)
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # 4h EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    ema_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d ADX trend strength filter (avoid strong trends, favor ranging)
    df_1d = get_htf_data(prices, '1d')
    # Calculate ADX (14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    # Smooth TR, DM+, DM-
    tr_smooth = pd.Series(tr).ewm(alpha=1/14, adjust=False).fillna(0).values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).fillna(0).values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).fillna(0).values
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / np.where(tr_smooth == 0, 1, tr_smooth)
    di_minus = 100 * dm_minus_smooth / np.where(tr_smooth == 0, 1, tr_smooth)
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) == 0, 1, (di_plus + di_minus))
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).fillna(0).values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    start_idx = 50  # Need sufficient history for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(ema_4h_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Mean reversion conditions
        near_lower_band = price <= bb_lower[i] * 1.01  # within 1% of lower band
        near_upper_band = price >= bb_upper[i] * 0.99  # within 1% of upper band
        
        # Trend filter: only take mean reversion in weak or ranging markets (ADX < 25)
        ranging_market = adx_aligned[i] < 25
        
        # Entry conditions
        if near_lower_band and volume_spike[i] and price > ema_4h_aligned[i] and ranging_market:
            # Oversold in uptrend context -> long
            signals[i] = 0.20
        elif near_upper_band and volume_spike[i] and price < ema_4h_aligned[i] and ranging_market:
            # Overbought in downtrend context -> short
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_Bollinger_Bands_Mean_Reversion_with_Volume_and_Trend_Filter_v1"
timeframe = "1h"
leverage = 1.0