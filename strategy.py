#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index regime filter with 1d ADX trend filter and 1h Donchian breakout for entry timing
# Uses 12h timeframe to target 50-150 total trades over 4 years (12-37/year)
# Choppiness Index (12h) > 61.8 = range (mean reversion), < 38.2 = trending (trend follow)
# ADX (1d) > 25 confirms trend strength for trend-following mode
# Entry timing: 1h Donchian(20) breakout in direction of higher timeframe trend
# Stoploss: ATR-based exit when price moves 2*ATR against position
# Designed to work in both bull (trend follow) and bear (mean reversion) markets via regime filter

name = "12h_chop_adx_1h_donchian_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h Choppiness Index (14-period)
    atr_12h = np.abs(high - low)
    tr_12h = np.maximum(np.abs(high - np.roll(low, 1)), np.absolute(np.roll(high, 1) - low))
    tr_12h[0] = atr_12h[0]  # first value
    atr_ma = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_ma * 14 / (highest_high - lowest_low)) / np.log10(14)
    chop = np.where((highest_high - lowest_low) == 0, 50, chop)  # avoid division by zero
    
    # 1d ADX (14-period) for trend strength
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and Directional Movement
    tr_1d = np.maximum(np.abs(high_1d - np.roll(low_1d, 1)), np.absolute(np.roll(high_1d, 1) - low_1d))
    tr_1d[0] = np.abs(high_1d[0] - low_1d[0])
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    atr_1d = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr_1d
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr_1d
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 1h Donchian Channel (20-period) for entry timing
    df_1h = get_htf_data(prices, '1h')
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    highest_high_1h = pd.Series(high_1h).rolling(window=20, min_periods=20).max().values
    lowest_low_1h = pd.Series(low_1h).rolling(window=20, min_periods=20).min().values
    donchian_upper_1h = align_htf_to_ltf(prices, df_1h, highest_high_1h)
    donchian_lower_1h = align_htf_to_ltf(prices, df_1h, lowest_low_1h)
    
    # 12h ATR for stoploss
    atr_12h = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if np.isnan(chop[i]) or np.isnan(adx_1d_aligned[i]) or np.isnan(donchian_upper_1h[i]) or np.isnan(donchian_lower_1h[i]) or np.isnan(atr_12h[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss: 2*ATR against position
        if position == 1:  # long position
            if close[i] < entry_price - 2.0 * atr_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] > entry_price + 2.0 * atr_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Determine regime based on 12h Choppiness Index
            # CHOP > 61.8 = range (mean reversion), CHOP < 38.2 = trending (trend follow)
            if chop[i] > 61.8:  # Range regime - mean reversion
                # Look for mean reversion entries at Donchian extremes
                # Long when price touches lower band and ADX shows weak trend (<25)
                if close[i] <= donchian_lower_1h[i] and adx_1d_aligned[i] < 25:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short when price touches upper band and ADX shows weak trend (<25)
                elif close[i] >= donchian_upper_1h[i] and adx_1d_aligned[i] < 25:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:  # Trending regime - trend follow
                # Look for trend continuation entries on breakouts
                # Long when price breaks above upper band and ADX shows strong trend (>25)
                if close[i] > donchian_upper_1h[i] and adx_1d_aligned[i] > 25:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short when price breaks below lower band and ADX shows strong trend (>25)
                elif close[i] < donchian_lower_1h[i] and adx_1d_aligned[i] > 25:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals