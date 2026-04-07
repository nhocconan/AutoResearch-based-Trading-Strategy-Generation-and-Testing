#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Choppiness Index regime filter with daily Donchian(20) breakout and volume confirmation
# Long when price breaks above daily Donchian high + volume > 1.5x 20-period average + weekly Choppiness < 38.2 (trending)
# Short when price breaks below daily Donchian low + volume > 1.5x 20-period average + weekly Choppiness < 38.2 (trending)
# Exit when price crosses 8-period EMA in opposite direction
# Stoploss at 2.5 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses daily Donchian for structure, weekly Choppiness for regime filter, and volume for confirmation
# Target: 50-150 total trades over 4 years (12-37/year)

name = "12h_chop_regime_donchian_vol_v1"
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
    
    # Daily data for Donchian breakout and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Weekly data for Choppiness Index regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Daily volume average (20-period)
    volume_1d = df_1d['volume'].values
    volume_1d_s = pd.Series(volume_1d)
    volume_ma = volume_1d_s.rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma)
    
    # Daily Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    highest_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    highest_high_aligned = align_htf_to_ltf(prices, df_1d, highest_high)
    lowest_low_aligned = align_htf_to_ltf(prices, df_1d, lowest_low)
    
    # Weekly Choppiness Index (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of True Range over 14 periods
    tr_sum = pd.Series(tr_1w).rolling(window=14, min_periods=14).sum().values
    
    # Sum of absolute price changes over 14 periods
    high_low_diff = np.abs(high_1w - low_1w)
    close_change = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    sum_high_low = pd.Series(high_low_diff).rolling(window=14, min_periods=14).sum().values
    sum_close_change = pd.Series(close_change).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index: 100 * log10(tr_sum / (sum_high_low + sum_close_change)) / log10(14)
    # Avoid division by zero
    denominator = sum_high_low + sum_close_change
    chop = np.where(denominator > 0, 100 * np.log10(tr_sum / denominator) / np.log10(14), 50)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # 8-period EMA for exit
    ema_8 = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    
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
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(highest_high_aligned[i]) or np.isnan(lowest_low_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(ema_8[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses below 8-period EMA
            elif close[i] < ema_8[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses above 8-period EMA
            elif close[i] > ema_8[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with volume confirmation and trending regime
            # Volume filter: volume > 1.5x 20-period average
            volume_filter = volume[i] > 1.5 * volume_ma_aligned[i]
            # Regime filter: weekly Choppiness < 38.2 (trending market)
            regime_filter = chop_aligned[i] < 38.2
            
            # Long: price breaks above daily Donchian high + volume filter + trending regime
            if close[i] > highest_high_aligned[i] and volume_filter and regime_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below daily Donchian low + volume filter + trending regime
            elif close[i] < lowest_low_aligned[i] and volume_filter and regime_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals