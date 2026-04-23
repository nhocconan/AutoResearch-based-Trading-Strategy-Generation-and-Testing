#!/usr/bin/env python3
"""
Hypothesis: 1h 4h Donchian breakout with 1d trend filter and volume confirmation.
Long when price breaks above 4h Donchian upper (20) AND 1d close > 1d EMA50 AND volume > 1.5x average.
Short when price breaks below 4h Donchian lower (20) AND 1d close < 1d EMA50 AND volume > 1.5x average.
Exit when price crosses 4h Donchian midpoint (mean reversion).
Uses discrete position sizing (0.20) to minimize fee churn. Targets 15-37 trades/year per symbol.
4h trend filter avoids counter-trend trades, Donchian provides structure, volume confirms conviction.
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
    
    # Load 4h data for Donchian channels - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Donchian channels (20-period) on 4h data
    def rolling_max(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).max().values
    
    def rolling_min(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).min().values
    
    donchian_h_4h = rolling_max(high_4h, 20)
    donchian_l_4h = rolling_min(low_4h, 20)
    donchian_m_4h = (donchian_h_4h + donchian_l_4h) / 2.0
    
    # Align 4h indicators to 1h timeframe
    donchian_h_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_h_4h)
    donchian_l_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_l_4h)
    donchian_m_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_m_4h)
    
    # Load 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d data
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA to 1h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(donchian_h_4h_aligned[i]) or np.isnan(donchian_l_4h_aligned[i]) or
            np.isnan(donchian_m_4h_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(vol_ma[i]) or not (8 <= hours[i] <= 20)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        daily_trend_up = close[i] > ema50_1d_aligned[i]
        daily_trend_down = close[i] < ema50_1d_aligned[i]
        
        vol_current = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: Price breaks above 4h Donchian upper AND daily uptrend AND volume confirmation
            if (high[i] > donchian_h_4h_aligned[i] and daily_trend_up and 
                vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below 4h Donchian lower AND daily downtrend AND volume confirmation
            elif (low[i] < donchian_l_4h_aligned[i] and daily_trend_down and 
                  vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below 4h Donchian midpoint (mean reversion)
                if close[i] < donchian_m_4h_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above 4h Donchian midpoint (mean reversion)
                if close[i] > donchian_m_4h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_4hDonchian20_1dEMA50_Volume"
timeframe = "1h"
leverage = 1.0