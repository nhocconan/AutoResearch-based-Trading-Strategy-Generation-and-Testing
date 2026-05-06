#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above 4h Donchian upper band AND price > 1d EMA34 AND volume > 2.0 * 20-period avg volume
# Short when price breaks below 4h Donchian lower band AND price < 1d EMA34 AND volume > 2.0 * 20-period avg volume
# Exit with ATR-based trailing stop: signal→0 when long and price < highest_high - 2.5 * ATR OR short and price < lowest_low + 2.5 * ATR
# Uses discrete sizing 0.25 to control drawdown in bear markets (BTC -77% in 2022 → ~19% loss at 0.25 exposure)
# Target: 80-150 total trades over 4 years (20-38/year) for 4h timeframe
# Donchian provides structure, 1d EMA34 filters primary trend, volume confirms breakout strength, ATR stop manages risk

name = "4h_Donchian20_1dEMA34_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Get 4h data ONCE before loop for Donchian channels and ATR
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian(20) channels
    high_4h_series = pd.Series(high_4h)
    low_4h_series = pd.Series(low_4h)
    donchian_upper = high_4h_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_4h_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h ATR(14) for stoploss
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    # Align HTF indicators to 4h timeframe (wait for completed HTF bar)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    atr_14_aligned = align_htf_to_ltf(prices, df_4h, atr_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    for i in range(60, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper with uptrend and volume spike
            if (close[i] > donchian_upper_aligned[i] and close[i-1] <= donchian_upper_aligned[i-1] and 
                close[i] > ema_34_1d_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = close[i]
            # Short: price breaks below Donchian lower with downtrend and volume spike
            elif (close[i] < donchian_lower_aligned[i] and close[i-1] >= donchian_lower_aligned[i-1] and 
                  close[i] < ema_34_1d_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = close[i]
        elif position == 1:
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, close[i])
            # Exit long: price drops below highest_high - 2.5 * ATR (trailing stop)
            if close[i] < highest_high_since_entry - 2.5 * atr_14_aligned[i]:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, close[i])
            # Exit short: price rises above lowest_low + 2.5 * ATR (trailing stop)
            if close[i] > lowest_low_since_entry + 2.5 * atr_14_aligned[i]:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals