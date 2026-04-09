#!/usr/bin/env python3
# 12h_camarilla_pivot_volume_regime_v3
# Hypothesis: 12h strategy using daily Camarilla pivot levels with volume confirmation and choppiness regime filter.
# Long when price touches Camarilla S3 level with volume > 1.5x 20-period average and chop < 61.8 (trending).
# Short when price touches Camarilla R3 level with volume > 1.5x 20-period average and chop < 61.8 (trending).
# Exit when price moves back to Camarilla H4/L4 levels or chop > 61.8 (ranging).
# Uses discrete position sizing (0.25) to minimize fee churn.
# Target: 12-37 trades/year (50-150 total over 4 years) on BTC/ETH/SOL to avoid overtrading and fee drag.
# Works in both bull and bear markets: Camarilla levels provide intraday support/resistance, volume confirms conviction, chop filter avoids whipsaws in ranging markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_volume_regime_v3"
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
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get daily HTF data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from daily data
    # Using previous day's OHLC to avoid look-ahead
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels calculation
    # H4 = close + 1.5 * (high - low)
    # L4 = close - 1.5 * (high - low)
    # H3 = close + 1.125 * (high - low)
    # L3 = close - 1.125 * (high - low)
    # H2 = close + 0.75 * (high - low)
    # L2 = close - 0.75 * (high - low)
    # H1 = close + 0.5 * (high - low)
    # L1 = close - 0.5 * (high - low)
    # Pivot = (high + low + close) / 3
    
    high_low_range = prev_high - prev_low
    camarilla_h4 = prev_close + 1.5 * high_low_range
    camarilla_l4 = prev_close - 1.5 * high_low_range
    camarilla_h3 = prev_close + 1.125 * high_low_range
    camarilla_l3 = prev_close - 1.125 * high_low_range
    camarilla_h2 = prev_close + 0.75 * high_low_range
    camarilla_l2 = prev_close - 0.75 * high_low_range
    camarilla_h1 = prev_close + 0.5 * high_low_range
    camarilla_l1 = prev_close - 0.5 * high_low_range
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3
    
    # Align HTF Camarilla levels to LTF (12h)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
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
    # Avoid division by zero
    denominator = highest_high - lowest_low
    denominator = np.where(denominator == 0, 1e-10, denominator)
    chop = 100 * np.log10(atr_sum / np.log10(atr_period) / denominator)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(chop[i]) or
            np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        # Regime filter: chop < 61.8 indicates trending market
        trending_market = chop[i] < 61.8
        
        if position == 1:  # Long position
            # Exit: price moves back to H4 level or chop > 61.8 (ranging market)
            if close[i] >= h4_aligned[i] or chop[i] >= 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price moves back to L4 level or chop > 61.8 (ranging market)
            if close[i] <= l4_aligned[i] or chop[i] >= 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for Camarilla level touches with volume and regime confirmation
            # Long: price touches or goes below L3 level
            # Short: price touches or goes above H3 level
            long_touch = close[i] <= l3_aligned[i]
            short_touch = close[i] >= h3_aligned[i]
            
            if long_touch and volume_confirmed and trending_market:
                position = 1
                signals[i] = 0.25
            elif short_touch and volume_confirmed and trending_market:
                position = -1
                signals[i] = -0.25
    
    return signals