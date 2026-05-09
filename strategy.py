#!/usr/bin/env python3
# Hypothesis: 4h Donchian breakout with 1d ATR-based volatility filter and volume confirmation
# Long when price breaks above Donchian upper band, volatility above threshold, and volume spike
# Short when price breaks below Donchian lower band, volatility above threshold, and volume spike
# Exit when price returns to Donchian middle line or reverses to opposite band
# Donchian channels provide clear breakout levels, volatility filter avoids choppy markets, volume adds conviction
# Designed for 4-6 trades per month per symbol (50-80/year) to minimize fee drag
# Works in both bull (breakouts continue) and bear (mean reversion to middle) markets

name = "4h_DonchianBreakout_VolatilityFilter_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1-day ATR for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # True Range calculation for ATR
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean().values
    
    # ATR ratio: current ATR / 50-period average ATR (volatility regime filter)
    atr_ma_50 = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_1d / atr_ma_50
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Donchian channels (20-period) on daily timeframe
    donch_high_20 = df_1d['high'].rolling(window=20, min_periods=20).max().values
    donch_low_20 = df_1d['low'].rolling(window=20, min_periods=20).min().values
    donch_mid_20 = (donch_high_20 + donch_low_20) / 2
    
    # Align Donchian levels to 4h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    donch_mid_aligned = align_htf_to_ltf(prices, df_1d, donch_mid_20)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for ATR and Donchian calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(donch_mid_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Volatility filter: only trade when volatility is elevated (ATR ratio > 1.2)
            if atr_ratio_aligned[i] > 1.2:
                # Enter long: price breaks above Donchian upper band with volume spike
                if close[i] > donch_high_aligned[i] and vol_confirm[i]:
                    signals[i] = 0.25
                    position = 1
                # Enter short: price breaks below Donchian lower band with volume spike
                elif close[i] < donch_low_aligned[i] and vol_confirm[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: price returns to middle line or reverses to lower band
            if close[i] <= donch_mid_aligned[i] or close[i] < donch_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to middle line or reverses to upper band
            if close[i] >= donch_mid_aligned[i] or close[i] > donch_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals