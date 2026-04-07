#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian channel breakout with 12-hour volume confirmation and daily volatility regime filter
# Long when price breaks above 4h Donchian(20) high + 12h volume > 1.5x 20-period average + daily ATR ratio > 1.2 (volatile regime)
# Short when price breaks below 4h Donchian(20) low + 12h volume > 1.5x 20-period average + daily ATR ratio > 1.2
# Exit when price crosses Donchian midpoint or ATR ratio < 0.8 (low volatility)
# Stoploss at 2.0 * ATR(14) from entry
# Position size: 0.25 (25% of capital)
# Target: 100-200 total trades over 4 years (25-50/year)

name = "4h_donchian20_12h_vol_volatility_regime_v1"
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
    
    # 12-hour data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Daily data for volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h Donchian Channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # 12h volume average (20-period)
    vol_12h = df_12h['volume'].values
    vol_12h_series = pd.Series(vol_12h)
    vol_ma_20 = vol_12h_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = vol_12h / (vol_ma_20 + 1e-10)
    
    # Daily ATR for volatility regime (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Daily ATR ratio (current ATR / 20-period average ATR)
    atr_1d_series = pd.Series(atr_1d)
    atr_ma_20 = atr_1d_series.rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr_1d / (atr_ma_20 + 1e-10)
    
    # Align 12h and daily data to 4h timeframe
    vol_ratio_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)  # Using 12h for alignment stability
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_12h, donchian_mid)
    
    # 4h ATR for stoploss (14-period)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ratio_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or 
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
            # Exit: price crosses Donchian midpoint or low volatility regime
            elif close[i] > donchian_mid_aligned[i] or atr_ratio_aligned[i] < 0.8:
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
            elif close[i] < donchian_mid_aligned[i] or atr_ratio_aligned[i] < 0.8:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with volume confirmation and volatility regime
            # Volatility filter: ATR ratio > 1.2 (volatile enough for breakout)
            volatile_regime = atr_ratio_aligned[i] > 1.2
            volume_confirm = vol_ratio_aligned[i] > 1.5
            
            # Long: price breaks above Donchian high + volume confirmation + volatile regime
            if close[i] > donchian_high_aligned[i] and volume_confirm and volatile_regime:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low + volume confirmation + volatile regime
            elif close[i] < donchian_low_aligned[i] and volume_confirm and volatile_regime:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals