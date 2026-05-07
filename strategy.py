#!/usr/bin/env python3
"""
1h_4h_1d_Structure_Filter_v1
Hypothesis: Combines 4h Donchian breakout for trend direction with 1d volume confirmation
and 1h momentum filter to avoid whipsaws. Uses timeframe hierarchy: 4h/1d for signal direction,
1h only for entry timing. Targets 15-35 trades/year to minimize fee drag while capturing
trend moves in both bull and bear markets through structured breakouts.
"""

name = "1h_4h_1d_Structure_Filter_v1"
timeframe = "1h"
leverage = 1.0

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
    
    # 4h Donchian channels (20-period) for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # 1d volume confirmation: volume > 20-period average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 1h momentum filter: RSI(14) to avoid overextended entries
    close_series = pd.Series(close)
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any critical value is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if position == 0 and in_session:
            # Long: price breaks above 4h Donchian high with volume confirmation and RSI not overbought
            if (close[i] > donchian_high_aligned[i] and 
                volume[i] > vol_ma_1d_aligned[i] and 
                rsi[i] < 70):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Donchian low with volume confirmation and RSI not oversold
            elif (close[i] < donchian_low_aligned[i] and 
                  volume[i] > vol_ma_1d_aligned[i] and 
                  rsi[i] > 30):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: price breaks below 4h Donchian low OR RSI becomes oversold
            if close[i] < donchian_low_aligned[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: price breaks above 4h Donchian high OR RSI becomes overbought
            if close[i] > donchian_high_aligned[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Outside session: maintain position or stay flat
            if position == 0:
                signals[i] = 0.0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals