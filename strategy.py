#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR-based trend filter.
Long when price breaks above Donchian upper band AND volume > 1.8x 20-period average AND ATR(14) > ATR(50) (volatile trending market).
Short when price breaks below Donchian lower band AND volume > 1.8x 20-period average AND ATR(14) > ATR(50).
Exit when price crosses the 20-period EMA in opposite direction.
Designed for low trade frequency (19-50/year) to minimize fee drag while capturing strong breakouts in both bull and bear markets.
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
    
    # Get 4h data for Donchian calculation (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate ATR(14) and ATR(50) on 4h for trend filter
    # True Range
    tr1 = np.abs(high_4h - low_4h)
    tr2 = np.abs(np.roll(high_4h, 1) - np.roll(close_4h, 1))
    tr3 = np.abs(np.roll(low_4h, 1) - np.roll(close_4h, 1))
    tr1[0] = high_4h[0] - low_4h[0]  # first bar
    tr2[0] = np.abs(high_4h[0] - close_4h[0])
    tr3[0] = np.abs(low_4h[0] - close_4h[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR calculations
    tr_series = pd.Series(tr)
    atr_14 = tr_series.ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_50 = tr_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Donchian channels (20-period) on 4h
    high_4h_series = pd.Series(high_4h)
    low_4h_series = pd.Series(low_4h)
    donchian_upper = high_4h_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_4h_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period EMA for exit
    close_4h_series = pd.Series(close_4h)
    ema_20 = close_4h_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate volume average (20-period) on 4h
    volume_4h_series = pd.Series(volume_4h)
    volume_ma_4h = volume_4h_series.rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 4h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_4h, atr_14)
    atr_50_aligned = align_htf_to_ltf(prices, df_4h, atr_50)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    ema_20_aligned = align_htf_to_ltf(prices, df_4h, ema_20)
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr_14_aligned[i]) or np.isnan(atr_50_aligned[i]) or 
            np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(ema_20_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        atr14 = atr_14_aligned[i]
        atr50 = atr_50_aligned[i]
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        ema20 = ema_20_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        high_price = high[i]
        low_price = low[i]
        
        # Trend filter: only trade in volatile markets (ATR14 > ATR50)
        volatile_market = atr14 > atr50
        
        if position == 0:
            # Long: price breaks above upper band AND volume > 1.8x avg AND volatile market
            if high_price > upper and vol > 1.8 * vol_ma and volatile_market:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band AND volume > 1.8x avg AND volatile market
            elif low_price < lower and vol > 1.8 * vol_ma and volatile_market:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below 20-period EMA
            if price < ema20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above 20-period EMA
            if price > ema20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Volume_ATRTrend_Filter"
timeframe = "4h"
leverage = 1.0