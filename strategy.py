#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation.
Long when price breaks above Donchian upper band AND weekly close > weekly EMA34 AND volume > 1.5x average.
Short when price breaks below Donchian lower band AND weekly close < weekly EMA34 AND volume > 1.5x average.
Exit when price crosses the weekly VWAP (mean reversion completion).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 15-25 trades/year per symbol.
Donchian provides clear structure, weekly trend filter avoids counter-trend trades in bear markets.
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
    
    # Load 1d data for Donchian calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Donchian(20) channels on 1d data
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe (already aligned as we're using 1d data)
    # But we need to shift by 1 to avoid look-ahead (use previous day's channel)
    donchian_upper = np.roll(donchian_upper, 1)
    donchian_lower = np.roll(donchian_lower, 1)
    donchian_upper[0] = donchian_upper[1] if len(donchian_upper) > 1 else donchian_upper[0]
    donchian_lower[0] = donchian_lower[1] if len(donchian_lower) > 1 else donchian_lower[0]
    
    # Load 1w data for trend filter and VWAP - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate EMA34 on 1w data
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate VWAP on 1w data (typical price * volume)
    typical_price_1w = (high_1w + low_1w + close_1w) / 3.0
    vwap_1w = (pd.Series(typical_price_1w * volume_1w).cumsum() / 
               pd.Series(volume_1w).cumsum()).values
    
    # Align 1w indicators to 1d timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w)
    
    # Volume average (20-period) on 1d timeframe
    vol_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vwap_1w_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Use 1d close for breakout detection
        daily_close = close_1d[i]
        daily_high = high_1d[i]
        daily_low = low_1d[i]
        
        weekly_trend_up = daily_close > ema34_1w_aligned[i]
        weekly_trend_down = daily_close < ema34_1w_aligned[i]
        
        vol_current = volume_1d[i]
        vol_ma_val = vol_ma_aligned[i]
        
        if position == 0:
            # Long: Price breaks above Donchian upper AND weekly uptrend AND volume confirmation
            if (daily_high > donchian_upper[i] and weekly_trend_up and 
                vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower AND weekly downtrend AND volume confirmation
            elif (daily_low < donchian_lower[i] and weekly_trend_down and 
                  vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below weekly VWAP (mean reversion complete)
                if daily_close < vwap_1w_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above weekly VWAP (mean reversion complete)
                if daily_close > vwap_1w_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian20_1wEMA34_VWAP"
timeframe = "1d"
leverage = 1.0