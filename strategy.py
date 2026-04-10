#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR volatility filter and 1w trend
# - Long when price breaks above Donchian upper (20) AND 1d ATR(14) > 1.5x ATR(50) AND 1w close > 1w open
# - Short when price breaks below Donchian lower (20) AND 1d ATR(14) > 1.5x ATR(50) AND 1w close < 1w open
# - Exit when price touches Donchian midpoint (mean reversion to equilibrium)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 19-50 trades/year on 4h timeframe (75-200 total over 4 years)
# - Donchian channels provide objective breakout levels; volatility filter ensures breakouts occur during expansion
# - Weekly trend filter ensures alignment with higher timeframe momentum, reducing counter-trend whipsaws

name = "4h_1d_1w_donchian_breakout_volatility_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Pre-compute Donchian channels (20-period) from 4h data
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Donchian upper/lower: highest high/lowest low over 20 periods
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Pre-compute 1d ATR volatility filter: ATR(14) > 1.5x ATR(50)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First period
    tr2[0] = np.abs(high_1d[0] - close_1d[0])  # First period
    tr3[0] = np.abs(low_1d[0] - close_1d[0])  # First period
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    vol_expansion = atr_14 > (1.5 * atr_50)
    vol_expansion_aligned = align_htf_to_ltf(prices, df_1d, vol_expansion)
    
    # Pre-compute 1w trend filter: bullish if close > open, bearish if close < open
    open_1w = df_1w['open'].values
    close_1w = df_1w['close'].values
    weekly_bullish = close_1w > open_1w
    weekly_bearish = close_1w < open_1w
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish)
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(vol_expansion_aligned[i]) or
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above Donchian upper AND volatility expansion AND weekly bullish
            if (prices['high'].iloc[i] > donchian_upper[i] and 
                vol_expansion_aligned[i] and 
                weekly_bullish_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below Donchian lower AND volatility expansion AND weekly bearish
            elif (prices['low'].iloc[i] < donchian_lower[i] and 
                  vol_expansion_aligned[i] and 
                  weekly_bearish_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to Donchian midpoint (mean reversion to equilibrium)
            # Exit when price touches Donchian midpoint
            exit_signal = False
            if position == 1:  # Long position
                if prices['low'].iloc[i] <= donchian_mid[i]:
                    exit_signal = True
            elif position == -1:  # Short position
                if prices['high'].iloc[i] >= donchian_mid[i]:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals