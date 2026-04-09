#!/usr/bin/env python3
# 12h_camarilla_pivot_volume_regime_v2
# Hypothesis: 12h strategy using 1d Camarilla pivot levels for mean reversion in ranging markets,
# volume confirmation for conviction, and choppiness filter to avoid trending markets.
# Long when price touches Camarilla S3 with volume > 1.5x 20-period average and chop > 61.8 (ranging).
# Short when price touches Camarilla R3 with volume > 1.5x 20-period average and chop > 61.8 (ranging).
# Exit when price reverts to Camarilla H3/L3 levels or chop < 38.2 (trending market).
# Uses discrete position sizing (0.25) to minimize fee churn.
# Target: 12-30 trades/year (50-120 total over 4 years) on BTC/ETH/SOL to avoid overtrading and fee drag.
# Works in both bull and bear markets: mean reversion in ranging markets (chop > 61.8) avoids losses in trends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_volume_regime_v2"
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
    
    # Get 1d HTF data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla uses previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values  # Shifted by 1 to use previous day
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla levels
    # H3/L3: (high - low) * 1.1/4 + close
    # S3/R3: (high - low) * 1.1/6 + close
    high_low_diff = prev_high - prev_low
    camarilla_h3 = prev_close + high_low_diff * 1.1 / 4
    camarilla_l3 = prev_close - high_low_diff * 1.1 / 4
    camarilla_s3 = prev_close - high_low_diff * 1.1 / 6
    camarilla_r3 = prev_close + high_low_diff * 1.1 / 6
    
    # Align HTF Camarilla levels to LTF (12h)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    
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
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        # Regime filter: chop > 61.8 indicates ranging market (good for mean reversion)
        ranging_market = chop[i] > 61.8
        # Exit regime: chop < 38.2 indicates trending market (exit positions)
        trending_market = chop[i] < 38.2
        
        if position == 1:  # Long position
            # Exit conditions: price reaches L3 or chop < 38.2 (trending)
            if close[i] <= l3_aligned[i] or trending_market:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: price reaches H3 or chop < 38.2 (trending)
            if close[i] >= h3_aligned[i] or trending_market:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry conditions: price touches S3/R3 with volume confirmation in ranging market
            long_entry = (close[i] <= s3_aligned[i]) and volume_confirmed and ranging_market
            short_entry = (close[i] >= r3_aligned[i]) and volume_confirmed and ranging_market
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals