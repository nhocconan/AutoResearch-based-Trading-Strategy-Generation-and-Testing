#!/usr/bin/env python3
# 12h_donchian_breakout_1w_volume_chop_v1
# Hypothesis: 12h Donchian(20) breakout with 1w trend filter (price vs 1w EMA50) and volume confirmation + chop regime filter.
# Works in bull/bear: 1w EMA50 acts as dynamic support/resistance; Donchian breakout captures momentum; volume ensures validity; chop filter avoids whipsaws in ranging markets.
# Target: 12-37 trades/year (50-150 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1w_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need 50 for EMA
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 12h Donchian(20) channels
    period = 20
    donchian_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    donchian_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # 12h ATR(14) for volatility regime
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 12h Choppiness Index (CHOP) regime filter
    chop_period = 14
    atr_sum = pd.Series(tr).rolling(window=chop_period, min_periods=chop_period).sum().values
    hh = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    ll = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(chop_period)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(chop[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low OR trend turns bearish OR chop extreme (trending too strong)
            if close[i] < donchian_low[i] or close[i] < ema_50_1w_aligned[i] or chop[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high OR trend turns bullish OR chop extreme (trending too strong)
            if close[i] > donchian_high[i] or close[i] > ema_50_1w_aligned[i] or chop[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation and chop not extreme (avoid whipsaws in strong trends)
            volume_confirmed = volume[i] > 1.5 * volume_ma[i]
            chop_not_extreme = chop[i] > 38.2  # Avoid strong trending regimes
            
            if volume_confirmed and chop_not_extreme:
                # Long: price breaks above Donchian high AND above 1w EMA50 (uptrend)
                if close[i] > donchian_high[i] and close[i] > ema_50_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below Donchian low AND below 1w EMA50 (downtrend)
                elif close[i] < donchian_low[i] and close[i] < ema_50_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals