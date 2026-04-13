#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian(20) breakout with 1d ATR volume confirmation and chop regime filter
    # Long when price breaks above Donchian upper + 1d ATR(14)/ATR(50) > 1.2 + chop > 61.8 (range)
    # Short when price breaks below Donchian lower + 1d ATR(14)/ATR(50) > 1.2 + chop > 61.8 (range)
    # Exit when price returns to Donchian midpoint or opposite breakout level
    # Uses discrete position sizing (0.25) to minimize fee churn and manage drawdown
    # Target: 50-150 total trades over 4 years (~12-37/year) to avoid fee drag
    # ATR ratio filter ensures breakouts occur during expansion phases
    # Chop filter ensures we only trade in ranging markets where mean reversion works
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) and ATR(50) for volatility expansion filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First period
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR calculations with min_periods
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = np.divide(atr_14, atr_50, out=np.full_like(atr_14, np.nan), where=atr_50!=0)
    
    # Calculate 1d Chopiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / (n * log10(highest_high - lowest_low)))
    # CHOP > 61.8 = ranging market (good for mean reversion/breakout fade)
    # CHOP < 38.2 = trending market
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_max_min = highest_high - lowest_low
    # Avoid division by zero and log of zero/negative
    log_range = np.log10(np.maximum(range_max_min, 1e-10))
    chop = np.divide(100 * np.log10(atr_sum), log_range, out=np.full_like(atr_sum, np.nan), where=log_range!=0)
    # CHOP is bounded between 0 and 100, but we'll use > 61.8 for ranging
    
    # Align all 1d indicators to 12h
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 12h Donchian channels (20-period)
    # Use rolling window on the 12h data directly
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(atr_ratio_aligned[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume/volatility filter: 1d ATR ratio > 1.2 (expansion)
        vol_expansion = atr_ratio_aligned[i] > 1.2
        
        # Regime filter: Chop > 61.8 (ranging market)
        ranging_market = chop_aligned[i] > 61.8
        
        # Breakout conditions
        bullish_breakout = (close[i] > donchian_upper[i] and 
                           vol_expansion and 
                           ranging_market)
        bearish_breakout = (close[i] < donchian_lower[i] and 
                           vol_expansion and 
                           ranging_market)
        
        # Exit conditions: return to midpoint or opposite breakout level
        long_exit = close[i] < donchian_mid[i]
        short_exit = close[i] > donchian_mid[i]
        
        if bullish_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_donchian_breakout_atr_chop_v1"
timeframe = "12h"
leverage = 1.0