#!/usr/bin/env python3
# 12h_camarilla_pivot_volume_regime_v1
# Hypothesis: 12h strategy using Camarilla pivot levels from 1-day timeframe for support/resistance,
# volume confirmation for conviction, and choppiness regime filter to avoid ranging markets.
# Long when price breaks above Camarilla H3 level with volume > 1.5x 20-period average and chop < 61.8 (trending).
# Short when price breaks below Camarilla L3 level with volume > 1.5x 20-period average and chop < 61.8 (trending).
# Exit when price returns to Camarilla Pivot level (mean reversion to equilibrium).
# Uses discrete position sizing (0.25) to minimize fee churn.
# Target: 12-37 trades/year (50-150 total over 4 years) on BTC/ETH/SOL to avoid overtrading and fee drag.
# Works in both bull and bear markets: Camarilla levels adapt to volatility, volume confirms breakout strength,
# chop filter avoids false signals in ranging markets. 12h timeframe reduces noise and fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_volume_regime_v1"
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
    
    # Load 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from 1d OHLC
    # Camarilla: Pivot = (H+L+C)/3, Range = H-L
    # H4 = Pivot + 1.1*(H-L)/2, H3 = Pivot + 1.1*(H-L)/4, H2 = Pivot + 1.1*(H-L)/6, H1 = Pivot + 1.1*(H-L)/12
    # L1 = Pivot - 1.1*(H-L)/12, L2 = Pivot - 1.1*(H-L)/6, L3 = Pivot - 1.1*(H-L)/4, L4 = Pivot - 1.1*(H-L)/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    h3_1d = pivot_1d + 1.1 * range_1d / 4.0
    l3_1d = pivot_1d - 1.1 * range_1d / 4.0
    pivot_1d_val = pivot_1d  # for exit condition
    
    # Align HTF Camarilla levels to LTF (12h)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d_val)
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index regime filter (14-period)
    atr_period = 14
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr_series = pd.Series(tr)
    atr_series = tr_series.rolling(window=atr_period, min_periods=atr_period).mean()
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    highest_high = high_series.rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low = low_series.rolling(window=atr_period, min_periods=atr_period).min().values
    atr_sum = tr_series.rolling(window=atr_period, min_periods=atr_period).sum().values
    chop = 100 * np.log10(atr_sum / np.log10(atr_period) / (highest_high - lowest_low))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or 
            np.isnan(pivot_1d_aligned[i]) or np.isnan(volume_ma[i]) or
            np.isnan(chop[i]) or np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        # Regime filter: chop < 61.8 indicates trending market
        trending_market = chop[i] < 61.8
        
        if position == 1:  # Long position
            # Exit: price returns to Camarilla Pivot level (mean reversion)
            if close[i] <= pivot_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to Camarilla Pivot level (mean reversion)
            if close[i] >= pivot_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for Camarilla breakout with volume and regime confirmation
            bullish_breakout = (close[i] > h3_1d_aligned[i] and close[i-1] <= h3_1d_aligned[i-1]) and volume_confirmed and trending_market
            bearish_breakout = (close[i] < l3_1d_aligned[i] and close[i-1] >= l3_1d_aligned[i-1]) and volume_confirmed and trending_market
            
            if bullish_breakout:
                position = 1
                signals[i] = 0.25
            elif bearish_breakout:
                position = -1
                signals[i] = -0.25
    
    return signals