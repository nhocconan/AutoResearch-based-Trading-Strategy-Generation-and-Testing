#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
# Long when price breaks above 20-bar Donchian high in 12h uptrend with volume spike (>1.8x 20-period volume MA).
# Short when price breaks below 20-bar Donchian low in 12h downtrend with volume spike.
# Uses ATR-based stoploss (signal→0 when price moves against position by 2.5*ATR).
# Designed for 6h timeframe to achieve 50-150 total trades over 4 years (12-37/year).
# Donchian channels provide robust trend-following structure, 12h EMA50 ensures higher timeframe alignment,
# Volume spike confirms institutional participation. Works in both bull and bear markets by only trading
# with the 12h trend, avoiding counter-trend whipsaws during ranging periods.

name = "6h_Donchian20_12hEMA50_VolumeSpike_ATR"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR for stoploss (using primary timeframe)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Get 6h data for Donchian channel calculation
    df_6h = get_htf_data(prices, '6h')
    
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Calculate 6h Donchian channels (20-period high/low)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    donchian_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_6h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_6h, donchian_low)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike detection (20-period volume MA on primary timeframe)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.8)  # Volume at least 1.8x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # Track entry price for ATR-based stoploss
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        vol_spike = volume_spike[i]
        upper_channel = donchian_high_aligned[i]
        lower_channel = donchian_low_aligned[i]
        trend_up = close_val > ema_50_12h_aligned[i]   # 12h uptrend
        trend_down = close_val < ema_50_12h_aligned[i]  # 12h downtrend
        
        if position == 0:
            # Long: price breaks above Donchian high AND 12h uptrend AND volume spike
            if close_val > upper_channel and trend_up and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short: price breaks below Donchian low AND 12h downtrend AND volume spike
            elif close_val < lower_channel and trend_down and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit conditions for long
            exit_signal = False
            # Stoploss: price moves against position by 2.5*ATR
            if close_val < entry_price - 2.5 * atr[i]:
                exit_signal = True
            # Exit: price breaks below Donchian low
            elif close_val < lower_channel:
                exit_signal = True
            # Exit: 12h trend changes to downtrend
            elif not trend_up:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit conditions for short
            exit_signal = False
            # Stoploss: price moves against position by 2.5*ATR
            if close_val > entry_price + 2.5 * atr[i]:
                exit_signal = True
            # Exit: price breaks above Donchian high
            elif close_val > upper_channel:
                exit_signal = True
            # Exit: 12h trend changes to uptrend
            elif not trend_down:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals