#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Donchian(20) breakout + 1-day volatility regime (ATR ratio) + volume confirmation
# Long when price breaks above Donchian(20) high + 1-day ATR(10)/ATR(30) > 0.8 (rising volatility) + volume > 1.5x 20-period average
# Short when price breaks below Donchian(20) low + same volatility/volume conditions
# Exit when price crosses Donchian(10) midpoint or volatility drops (ATR ratio < 0.6)
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses 1-day ATR ratio for regime and volume for confirmation
# Target: 100-200 total trades over 4 years (25-50/year)

name = "6h_donchian20_1d_atr_ratio_vol_v1"
timeframe = "6h"
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
    
    # 1-day data for ATR ratio and regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1-day ATR(10) and ATR(30)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1-day
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr2_1d[0] = tr1_1d[0]
    tr3_1d[0] = tr1_1d[0]
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    
    atr10_1d = pd.Series(tr_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr30_1d = pd.Series(tr_1d).ewm(span=30, adjust=False, min_periods=30).mean().values
    atr_ratio_1d = atr10_1d / (atr30_1d + 1e-10)
    
    # Align ATR ratio to 6h
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # 6-hour Donchian channels
    donchian_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid_10 = (donchian_high_20 + donchian_low_20) / 2
    
    # 6-period ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or 
            np.isnan(donchian_mid_10[i]) or np.isnan(atr_ratio_1d_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_avg[i])):
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
            # Exit: price crosses Donchian midpoint or low volatility regime
            elif close[i] < donchian_mid_10[i] or atr_ratio_1d_aligned[i] < 0.6:
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
            # Exit: price crosses Donchian midpoint or low volatility regime
            elif close[i] > donchian_mid_10[i] or atr_ratio_1d_aligned[i] < 0.6:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with volatility and volume confirmation
            # Volatility filter: ATR ratio > 0.8 (rising volatility)
            vol_regime = atr_ratio_1d_aligned[i] > 0.8
            # Volume filter: current volume > 1.5x 20-period average
            vol_confirm = volume[i] > 1.5 * vol_avg[i]
            
            # Long: price breaks above Donchian(20) high + volatility + volume
            if close[i] > donchian_high_20[i] and vol_regime and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian(20) low + volatility + volume
            elif close[i] < donchian_low_20[i] and vol_regime and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals