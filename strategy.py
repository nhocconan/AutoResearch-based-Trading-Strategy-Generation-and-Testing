#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1-day volume confirmation + ATR volatility filter
# Long when price breaks above 4h Donchian upper channel + daily volume > 20-day average + ATR(14) > 0.5 * ATR(50)
# Short when price breaks below 4h Donchian lower channel + daily volume > 20-day average + ATR(14) > 0.5 * ATR(50)
# Exit when price crosses the 4h Donchian midline (average of upper/lower) or ATR volatility drops below threshold
# Designed to capture strong breakouts in both bull and bear markets with volume confirmation to avoid false signals
# Target: 75-200 total trades over 4 years (19-50/year)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for volume average
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    avg_vol_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    avg_vol_20_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_20)
    
    # Calculate 4h Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian upper and lower channels
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Calculate ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate long-term ATR (50-period) for dynamic threshold
    atr_long = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(avg_vol_20_aligned[i]) or np.isnan(atr[i]) or np.isnan(atr_long[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current daily volume > 20-day average
        vol_confirm = volume_1d[i // 16] > avg_vol_20_aligned[i] if i // 16 < len(volume_1d) else False
        
        # Volatility filter: ATR(14) > 0.5 * ATR(50)
        vol_filter = atr[i] > 0.5 * atr_long[i]
        
        price = close[i]
        
        if position == 0:
            # Enter long: price breaks above Donchian upper + volume confirmation + volatility filter
            long_signal = (price > donch_high[i]) and vol_confirm and vol_filter
            
            # Enter short: price breaks below Donchian lower + volume confirmation + volatility filter
            short_signal = (price < donch_low[i]) and vol_confirm and vol_filter
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Donchian midline OR volatility drops below threshold
            exit_signal = (price < donch_mid[i]) or (atr[i] <= 0.5 * atr_long[i])
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Donchian midline OR volatility drops below threshold
            exit_signal = (price > donch_mid[i]) or (atr[i] <= 0.5 * atr_long[i])
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeVolatilityFilter"
timeframe = "4h"
leverage = 1.0