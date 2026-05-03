#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter, volume spike confirmation, and ATR-based stoploss.
# Long when price breaks above Donchian(20) upper band in 1d uptrend with volume spike (>2x 20-period vol MA).
# Short when price breaks below Donchian(20) lower band in 1d downtrend with volume spike.
# Exit on opposite Donchian breakout or ATR(14) trailing stop (3*ATR from extreme).
# Uses discrete sizing 0.30 to balance return and drawdown. Target: 75-200 total trades over 4 years.
# Donchian channels provide structural breakouts, 1d EMA50 ensures higher timeframe alignment,
# Volume spike confirms institutional participation. Works in both bull and bear markets by only trading
# with the 1d trend, avoiding counter-trend whipsaws.

name = "4h_Donchian20_1dEMA50_VolumeSpike_ATR"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian and ATR calculation (primary timeframe data)
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    highest_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    
    # Calculate 4h ATR(14) for volatility and stoploss
    tr1 = pd.Series(high_4h).rolling(window=2, min_periods=1).max().values - pd.Series(low_4h).rolling(window=2, min_periods=1).min().values
    tr2 = abs(pd.Series(high_4h).rolling(window=2, min_periods=1).max().values - pd.Series(close).rolling(window=2, min_periods=1).shift(1).values)
    tr3 = abs(pd.Series(low_4h).rolling(window=2, min_periods=1).min().values - pd.Series(close).rolling(window=2, min_periods=1).shift(1).values)
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align Donchian and ATR to LTF (15m) - but since we're on 4h, we need to be careful
    # Actually, we're using 4h as primary timeframe, so we work directly with 4h data
    # But we need to align HTF (1d) data to 4h timeframe
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike detection (20-period volume MA on 4h data)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol_spike = volume_spike[i]
        upper_band = donchian_upper[i]
        lower_band = donchian_lower[i]
        trend_up = close_val > ema_50_1d_aligned[i]   # 1d uptrend
        trend_down = close_val < ema_50_1d_aligned[i]  # 1d downtrend
        atr_val = atr[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper band AND 1d uptrend AND volume spike
            if close_val > upper_band and trend_up and vol_spike:
                signals[i] = 0.30
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            # Short: price breaks below Donchian lower band AND 1d downtrend AND volume spike
            elif close_val < lower_band and trend_down and vol_spike:
                signals[i] = -0.30
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
        elif position == 1:
            # Update highest price since entry for trailing stop
            highest_since_entry = max(highest_since_entry, high_val)
            
            # Exit conditions:
            # 1. Price breaks below Donchian lower band (opposite breakout)
            # 2. ATR trailing stop: price drops 3*ATR from highest since entry
            if close_val < lower_band or close_val < (highest_since_entry - 3.0 * atr_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Update lowest price since entry for trailing stop
            lowest_since_entry = min(lowest_since_entry, low_val)
            
            # Exit conditions:
            # 1. Price breaks above Donchian upper band (opposite breakout)
            # 2. ATR trailing stop: price rises 3*ATR from lowest since entry
            if close_val > upper_band or close_val > (lowest_since_entry + 3.0 * atr_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals