#!/usr/bin/env python3
# 1d_donchian_breakout_weekly_trend_v1
# Hypothesis: Daily Donchian(20) breakouts filtered by weekly HMA(21) trend and volume confirmation.
# Long when price breaks above Donchian upper channel with price > weekly HMA and volume > 1.5x 20-day average.
# Short when price breaks below Donchian lower channel with price < weekly HMA and volume > 1.5x 20-day average.
# Exit when price returns to Donchian midpoint (mean reversion within channel).
# Uses discrete position sizing (0.25) to minimize fee churn.
# Target: 15-25 trades/year (60-100 total over 4 years) on BTC/ETH/SOL.
# Works in bull markets via breakout momentum and bear markets via short breakdowns.
# Weekly HMA filter ensures we only trade with the higher timeframe trend, reducing false breakouts.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_weekly_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-day)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Donchian Channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Weekly HMA(21) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    # HMA calculation: WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='full')[-len(values):] * weights.sum() / (weights * np.arange(1, len(values) + 1)).sum()
    n_hma = 21
    half_n = n_hma // 2
    sqrt_n = int(np.sqrt(n_hma))
    wma_half = np.array([wma(close_1w[i - half_n + 1:i + 1], half_n) if i >= half_n - 1 else np.nan for i in range(len(close_1w))])
    wma_full = np.array([wma(close_1w[i - n_hma + 1:i + 1], n_hma) if i >= n_hma - 1 else np.nan for i in range(len(close_1w))])
    raw_hma = 2 * wma_half - wma_full
    hma_1w = np.array([wma(raw_hma[i - sqrt_n + 1:i + 1], sqrt_n) if i >= sqrt_n - 1 else np.nan for i in range(len(raw_hma))])
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(hma_1w_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price returns to Donchian midpoint
            if close[i] <= donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to Donchian midpoint
            if close[i] >= donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for Donchian breakout with weekly trend and volume confirmation
            bullish_breakout = (close[i] > donchian_high[i] and 
                               close[i] > hma_1w_aligned[i] and 
                               volume_confirmed)
            bearish_breakout = (close[i] < donchian_low[i] and 
                               close[i] < hma_1w_aligned[i] and 
                               volume_confirmed)
            
            if bullish_breakout:
                position = 1
                signals[i] = 0.25
            elif bearish_breakout:
                position = -1
                signals[i] = -0.25
    
    return signals