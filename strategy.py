#!/usr/bin/env python3
# 4h_donchian_breakout_volume_chop_v1
# Hypothesis: 4h Donchian channel breakout with volume confirmation and choppiness regime filter.
# Long when price breaks above Donchian(20) high with volume > 1.5x average and CHOP > 61.8 (rangy).
# Short when price breaks below Donchian(20) low with volume > 1.5x average and CHOP > 61.8.
# Exit on opposite Donchian break or when CHOP < 38.2 (trending).
# Uses 1d trend filter: only take longs if price > daily EMA50, shorts if price < daily EMA50.
# Designed to work in both bull (breakouts) and bear (rangy markets with mean reversion).
# Target: 20-50 trades/year (80-200 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Choppiness Index (CHOP) - 14 period
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(pd.Series(atr).rolling(window=14, min_periods=14).sum().values / (max_high - min_low)) / np.log10(14)
    # Handle division by zero and invalid values
    chop = np.where((max_high - min_low) == 0, 50, chop)
    chop = np.where(np.isnan(chop), 50, chop)
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate daily EMA50
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(close[i]) or 
            np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Regime filter: only trade in choppy markets (CHOP > 61.8)
        choppy_market = chop[i] > 61.8
        
        # Trend filter from daily EMA50
        bullish_trend = close[i] > ema_50_aligned[i]
        bearish_trend = close[i] < ema_50_aligned[i]
        
        if position == 1:  # Long position
            # Exit: Price breaks below Donchian low OR market becomes trending (CHOP < 38.2)
            if close[i] < donchian_low[i] or chop[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price breaks above Donchian high OR market becomes trending (CHOP < 38.2)
            if close[i] > donchian_high[i] or chop[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry conditions: Donchian breakout with volume confirmation in choppy market
            bullish_breakout = (close[i] > donchian_high[i]) and volume_confirmed and choppy_market and bullish_trend
            bearish_breakout = (close[i] < donchian_low[i]) and volume_confirmed and choppy_market and bearish_trend
            
            if bullish_breakout:
                position = 1
                signals[i] = 0.25
            elif bearish_breakout:
                position = -1
                signals[i] = -0.25
    
    return signals