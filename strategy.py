#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian(20) breakout with 1-day ATR-based volatility filter and volume confirmation
# Long when price breaks above 12h Donchian upper + 1-day ATR > 20-period median (high volatility regime) + volume > 1.5x 20-period average
# Short when price breaks below 12h Donchian lower + 1-day ATR > 20-period median + volume > 1.5x 20-period average
# Exit when price crosses 12h Donchian middle or volatility drops below median
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Target: 75-150 total trades over 4 years (19-38/year)

name = "12h_donchian20_1d_atr_vol_filter_v1"
timeframe = "12h"
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
    
    # 1-day data for ATR volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day ATR(14) for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 20-period median of ATR for volatility regime filter
    atr_1d_series = pd.Series(atr_1d)
    atr_median = atr_1d_series.rolling(window=20, min_periods=20).median().values
    atr_median_aligned = align_htf_to_ltf(prices, df_1d, atr_median)
    
    # 12-hour Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or np.isnan(atr_median_aligned[i]) or 
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR(14) - calculate ATR for stoploss
            tr1 = high[i] - low[i]
            tr2 = np.abs(high[i] - close[i-1]) if i > 0 else tr1
            tr3 = np.abs(low[i] - close[i-1]) if i > 0 else tr1
            atr_now = max(tr1, max(tr2, tr3))
            if close[i] < entry_price - 2.0 * atr_now:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses Donchian middle or low volatility regime
            elif close[i] < donchian_middle[i] or atr_1d[i] < atr_median_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR(14)
            tr1 = high[i] - low[i]
            tr2 = np.abs(high[i] - close[i-1]) if i > 0 else tr1
            tr3 = np.abs(low[i] - close[i-1]) if i > 0 else tr1
            atr_now = max(tr1, max(tr2, tr3))
            if close[i] > entry_price + 2.0 * atr_now:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses Donchian middle or low volatility regime
            elif close[i] > donchian_middle[i] or atr_1d[i] < atr_median_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with volatility and volume filters
            high_vol_regime = atr_1d[i] > atr_median_aligned[i]
            vol_confirm = volume[i] > volume_threshold[i]
            
            # Long: break above Donchian upper + high volatility + volume confirmation
            if close[i] > donchian_upper[i] and high_vol_regime and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: break below Donchian lower + high volatility + volume confirmation
            elif close[i] < donchian_lower[i] and high_vol_regime and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals