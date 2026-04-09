#!/usr/bin/env python3
# 1d_donchian_breakout_volume_chop_v1
# Hypothesis: 1d strategy using Donchian(20) breakouts with volume confirmation (>1.5x 20-bar avg volume) and choppiness regime filter (CHOP(14) > 61.8 for ranging markets). Enters long when price breaks above Donchian upper channel with volume confirmation and chop > 61.8 (mean reversion setup); enters short when price breaks below Donchian lower channel with volume confirmation and chop > 61.8. Uses weekly EMA(50) from 1w HTF for trend filter: only long when price > weekly EMA(50), only short when price < weekly EMA(50). Exits on opposite Donchian channel touch or close beyond Donchian(25) channels. Uses discrete sizing (0.25) to limit fee churn. Target: 7-25 trades/year (30-100 total over 4 years). Donchian breakouts capture momentum; volume confirms conviction; chop filter ensures ranging conditions where mean reversion works; weekly EMA avoids counter-trend trades in strong trends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_volume_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Donchian channels (20-period)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_upper = high_s.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_s.rolling(window=20, min_periods=20).min().values
    donchian_upper_25 = high_s.rolling(window=25, min_periods=25).max().values  # for stop
    donchian_lower_25 = low_s.rolling(window=25, min_periods=25).min().values   # for stop
    
    # Choppiness Index (14-period)
    def choppiness_index(high, low, close, window=14):
        atr = pd.Series(high).rolling(window).max() - pd.Series(low).rolling(window).min()
        atr_sum = atr.rolling(window=window, min_periods=window).sum()
        close_range = pd.Series(close).rolling(window=window, min_periods=window).max() - \
                     pd.Series(close).rolling(window=window, min_periods=window).min()
        chop = 100 * np.log10(atr_sum / close_range) / np.log10(window)
        return chop.values
    
    chop = choppiness_index(high, low, close, 14)
    
    # Multi-timeframe: weekly EMA(50) trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    close_1w_s = pd.Series(close_1w)
    ema_50_1w = close_1w_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(chop[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(donchian_upper_25[i]) or np.isnan(donchian_lower_25[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Choppiness regime: CHOP > 61.8 indicates ranging market (good for mean reversion)
        chop_regime = chop[i] > 61.8
        
        # Trend filters from weekly EMA
        uptrend_filter = close[i] > ema_50_1w_aligned[i]
        downtrend_filter = close[i] < ema_50_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price touches Donchian lower channel or breaks below Donchian(25) lower (failed breakout)
            if close[i] <= donchian_lower[i] or close[i] < donchian_lower_25[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches Donchian upper channel or breaks above Donchian(25) upper (failed breakout)
            if close[i] >= donchian_upper[i] or close[i] > donchian_upper_25[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for Donchian breakout with volume confirmation, chop regime, and trend alignment
            bullish_breakout = (close[i] > donchian_upper[i]) and volume_confirmed and chop_regime and not uptrend_filter
            bearish_breakout = (close[i] < donchian_lower[i]) and volume_confirmed and chop_regime and not downtrend_filter
            
            if bullish_breakout:
                position = 1
                signals[i] = 0.25
            elif bearish_breakout:
                position = -1
                signals[i] = -0.25
    
    return signals