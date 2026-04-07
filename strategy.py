#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian breakout with 1-day EMA trend filter and volume confirmation
# Long when price breaks above 4h Donchian(20) upper + price > 1-day EMA(50) + volume > 1.5x avg volume
# Short when price breaks below 4h Donchian(20) lower + price < 1-day EMA(50) + volume > 1.5x avg volume
# Exit when price crosses Donchian midline or trend reverses
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses 1-day EMA for trend filter and 4h Donchian for breakout signals
# Target: 75-200 total trades over 4 years (19-50/year)

name = "4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day EMA(50)
    close_1d = df_1d['close'].values
    close_1d_s = pd.Series(close_1d)
    ema_50 = close_1d_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # 4-hour Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Volume confirmation: 1.5x average volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (volume_ma + 1e-10)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_ratio[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses Donchian midline or trend reverses (price < EMA)
            elif close[i] < donchian_middle[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses Donchian midline or trend reverses (price > EMA)
            elif close[i] > donchian_middle[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for breakout entries with volume confirmation and trend filter
            # Volume filter: volume > 1.5x average volume
            volume_confirm = volume_ratio[i] > 1.5
            
            # Long: break above Donchian upper + price > EMA + volume confirmation
            if close[i] > donchian_upper[i] and close[i] > ema_50_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: break below Donchian lower + price < EMA + volume confirmation
            elif close[i] < donchian_lower[i] and close[i] < ema_50_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals