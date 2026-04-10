#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume confirmation and 1w trend filter
# - Entry: Long when price breaks above 20-bar Donchian high + 1d volume > 2.0x 20-period average + 1w close > 1w open (bullish weekly candle)
#          Short when price breaks below 20-bar Donchian low + 1d volume > 2.0x 20-period average + 1w close < 1w open (bearish weekly candle)
# - Exit: Close-based reversal - exit long when price < 10-bar Donchian low, exit short when price > 10-bar Donchian high
# - Position sizing: 0.25 (discrete levels to minimize fee churn)
# - Uses Donchian channels for structure, volume for confirmation, 1w for trend filter
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within HARD MAX: 200 total
# - Designed for 4h timeframe to balance trade frequency and signal quality, with HTF filters for robustness

name = "4h_1d_1w_donchian_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Pre-compute 1d OHLC for volume
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Pre-compute 1w OHLC for trend filter
    open_1w = df_1w['open'].values
    close_1w = df_1w['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    high_10 = pd.Series(high_4h).rolling(window=10, min_periods=10).max().values
    low_10 = pd.Series(low_4h).rolling(window=10, min_periods=10).min().values
    
    # Calculate 1d volume moving average (20-period)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w bullish/bearish candle
    bullish_week = close_1w > open_1w  # True for bullish weekly candle
    bearish_week = close_1w < open_1w  # True for bearish weekly candle
    
    # Align all HTF data to 4h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)  # Dummy alignment - will be replaced
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)   # Dummy alignment - will be replaced
    high_10_aligned = align_htf_to_ltf(prices, df_1d, high_10)  # Dummy alignment - will be replaced
    low_10_aligned = align_htf_to_ltf(prices, df_1d, low_10)   # Dummy alignment - will be replaced
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    bullish_week_aligned = align_htf_to_ltf(prices, df_1w, bullish_week.astype(float))
    bearish_week_aligned = align_htf_to_ltf(prices, df_1w, bearish_week.astype(float))
    
    # Actually, Donchian is already on 4h, so no need to align - but we need to align the HTF filters
    # Re-align the 4h Donchian properly (it's already LTF)
    high_20_aligned = high_20
    low_20_aligned = low_20
    high_10_aligned = high_10
    low_10_aligned = low_10
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(high_10_aligned[i]) or np.isnan(low_10_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(bullish_week_aligned[i]) or 
            np.isnan(bearish_week_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 4h close
        close_price = close_4h[i]
        
        # Get current 1d volume for confirmation (need to align volume_1d)
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        volume_confirmation = volume_1d_aligned[i] > 2.0 * volume_ma_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above 20-bar Donchian high + volume confirmation + bullish weekly candle
            if (close_price > high_20_aligned[i] and 
                volume_confirmation and 
                bullish_week_aligned[i] > 0.5):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below 20-bar Donchian low + volume confirmation + bearish weekly candle
            elif (close_price < low_20_aligned[i] and 
                  volume_confirmation and 
                  bearish_week_aligned[i] > 0.5):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit long when price < 10-bar Donchian low
            # Exit short when price > 10-bar Donchian high
            if position == 1:
                if close_price < low_10_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close_price > high_10_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals