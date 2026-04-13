#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 1d/1w regime filter and volume confirmation.
    # 1d trend via EMA50 defines market regime (bull/bear). 1w EMA200 confirms higher-timeframe bias.
    # Donchian breakout captures momentum in direction of 1d trend. Volume confirms participation.
    # Target: 75-200 total trades over 4 years = 19-50/year.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for regime and volume (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for higher-timeframe bias (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend regime
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1w EMA200 for higher-timeframe bias
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 1d volume MA(20) for confirmation
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 4h Donchian channels (20-period)
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Align HTF indicators to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema200_1w_aligned[i]) or 
            np.isnan(volume_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: 1d price > EMA50 = bullish regime, < EMA50 = bearish regime
        bullish_regime = close[i] > ema50_1d_aligned[i]
        bearish_regime = close[i] < ema50_1d_aligned[i]
        
        # Higher-timeframe filter: 1w price > EMA200 = bullish bias, < EMA200 = bearish bias
        weekly_bullish_bias = close[i] > ema200_1w_aligned[i]
        weekly_bearish_bias = close[i] < ema200_1w_aligned[i]
        
        # Volume filter: current volume > 20-period EMA
        volume_filter = volume[i] > volume_ma_1d_aligned[i]
        
        # Donchian breakout conditions
        breakout_long = close[i] > donchian_high_aligned[i]  # Break above upper channel
        breakout_short = close[i] < donchian_low_aligned[i]  # Break below lower channel
        
        # Entry conditions: breakout in direction of 1d trend with 1w bias and volume confirmation
        long_entry = breakout_long and bullish_regime and weekly_bullish_bias and volume_filter
        short_entry = breakout_short and bearish_regime and weekly_bearish_bias and volume_filter
        
        # Exit conditions: price returns to opposite Donchian level
        long_exit = close[i] < donchian_low_aligned[i]
        short_exit = close[i] > donchian_high_aligned[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
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

name = "4h_1d_1w_donchian_breakout_regime_volume_v1"
timeframe = "4h"
leverage = 1.0