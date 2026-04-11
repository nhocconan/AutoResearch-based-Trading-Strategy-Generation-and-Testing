#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 1d weekly pivot direction + volume confirmation
# - Long: price breaks above 6h Donchian upper band with volume and 1d weekly pivot bias long
# - Short: price breaks below 6h Donchian lower band with volume and 1d weekly pivot bias short
# - Exit: opposite Donchian band touch or volatility contraction (ATR ratio < 0.8)
# - Uses 1d data for weekly pivot calculation (based on prior week OHLC) aligned to 6h
# - Weekly pivot bias: price above weekly pivot = long bias, below = short bias
# - Works in bull/bear markets by combining breakout momentum with HTF structure
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits

name = "6h_1d_donchian_weeklypivot_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for weekly pivot (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return signals
    
    # Pre-compute 1d weekly pivot levels (based on prior week OHLC)
    # Resample 1d to weekly: get weekly high, low, close
    weekly_high = df_1d['high'].rolling(window=7, min_periods=7).max().shift(1)  # Prior week
    weekly_low = df_1d['low'].rolling(window=7, min_periods=7).min().shift(1)
    weekly_close = df_1d['close'].rolling(window=7, min_periods=7).last().shift(1)
    
    # Weekly pivot calculation: P = (H + L + C) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot.values)
    
    # Pre-compute 6h Donchian channels (20-period)
    high_rolling_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_rolling_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 6h volume confirmation (20-period average)
    volume = prices['volume'].values
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute 6h ATR for volatility-based exit (14-period)
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(high_rolling_max[i]) or np.isnan(low_rolling_min[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(volume_sma_20[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        high_price = high[i]
        low_price = low[i]
        volume_current = volume[i]
        
        # Donchian bands
        donchian_upper = high_rolling_max[i]
        donchian_lower = low_rolling_min[i]
        
        # Weekly pivot bias
        weekly_pivot = weekly_pivot_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Volatility filter: avoid breakouts during low volatility (ATR ratio < 0.8 of 50-period average)
        atr_50_avg = pd.Series(atr).rolling(window=50, min_periods=50).mean().iloc[i] if i >= 50 else atr[i]
        vol_filter = atr[i] > 0.8 * atr_50_avg if not np.isnan(atr_50_avg) else True
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price breaks above Donchian upper with volume, pivot bias long, and volatility OK
        if close_price > donchian_upper and vol_confirm and weekly_pivot > close_price and vol_filter:
            enter_long = True
        
        # Short breakout: price breaks below Donchian lower with volume, pivot bias short, and volatility OK
        if close_price < donchian_lower and vol_confirm and weekly_pivot < close_price and vol_filter:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long: price touches Donchian lower band OR volatility contraction
            exit_long = close_price <= donchian_lower or (atr[i] < 0.6 * atr_50_avg and not np.isnan(atr_50_avg))
        elif position == -1:
            # Exit short: price touches Donchian upper band OR volatility contraction
            exit_short = close_price >= donchian_upper or (atr[i] < 0.6 * atr_50_avg and not np.isnan(atr_50_avg))
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals