#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian channel breakout with 1d EMA50 trend filter and volume confirmation on 1h timeframe.
Long when price breaks above 4h Donchian upper band AND volume > 1.3x 20-period average AND price > 1d EMA50.
Short when price breaks below 4h Donchian lower band AND volume > 1.3x 20-period average AND price < 1d EMA50.
Exit when price crosses the 1d EMA50 in opposite direction.
Donchian channels provide clear breakout levels, 1d EMA50 filters for higher timeframe trend,
volume confirmation reduces false breakouts. Designed to work in both bull and bear markets
by trading with the 1d trend while using 4h structure for entries.
Targets 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
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
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    high_4h_series = pd.Series(high_4h)
    low_4h_series = pd.Series(low_4h)
    donchian_upper = high_4h_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_4h_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA50
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 4h volume average (20-period)
    volume_4h_series = pd.Series(volume_4h)
    volume_ma_4h = volume_4h_series.rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 1h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for indicators (Donchian + EMA50)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        ema_50 = ema_50_1d_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        high_price = high[i]
        low_price = low[i]
        
        if position == 0:
            # Long: price breaks above 4h Donchian upper AND volume > 1.3x avg AND price > 1d EMA50 (bullish trend)
            if high_price > upper and vol > 1.3 * vol_ma and price > ema_50:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Donchian lower AND volume > 1.3x avg AND price < 1d EMA50 (bearish trend)
            elif low_price < lower and vol > 1.3 * vol_ma and price < ema_50:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below 1d EMA50
            if price < ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price crosses above 1d EMA50
            if price > ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_DonchianBreakout_4hChannels_1dEMA50_Volume"
timeframe = "1h"
leverage = 1.0