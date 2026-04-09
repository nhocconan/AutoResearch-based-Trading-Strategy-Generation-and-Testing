#!/usr/bin/env python3
# 4h_donchian_breakout_volume_chop_v1
# Hypothesis: 4h Donchian(20) breakout with volume confirmation and choppiness regime filter.
# Long when price breaks above Donchian upper band with volume > 1.5x average and CHOP > 61.8 (range).
# Short when price breaks below Donchian lower band with volume > 1.5x average and CHOP > 61.8.
# Exit on opposite Donchian break or when CHOP < 38.2 (trending regime).
# Uses 1d HTF for Donchian calculation to reduce noise and false breakouts.
# Designed to work in both bull (trend continuation) and bear (mean reversion in range) markets.
# Target: 20-50 trades/year (80-200 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_chop_v1"
timeframe = "4h"
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
    
    # Get 1d data for Donchian calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper band: highest high over 20 days
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low over 20 days
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe (completed 1d bars only)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate Choppiness Index (14-period) on 1d for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (max(high,n) - min(low,n))))
    # Simplified: CHOP = 100 * log10(ATR_sum / (log10(14) * (HHV - LLV)))
    tr_1d = np.maximum(
        np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1])),
        np.abs(low_1d[1:] - close_1d[:-1])
    )
    # Prepend first TR as high-low for simplicity
    tr_1d = np.concatenate([[high_1d[0] - low_1d[0]], tr_1d])
    atr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_sum = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    hhvl_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    llv_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    denominator = np.log10(14) * (hhvl_14 - llv_14)
    # Avoid division by zero
    denominator = np.where(denominator == 0, 1e-10, denominator)
    chop = 100 * np.log10(atr_sum / denominator)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(volume_ma[i]) or
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Regime filter: only trade in ranging markets (CHOP > 61.8)
        ranging_market = chop_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: Price breaks below Donchian lower band OR market becomes trending
            if close[i] < donchian_low_aligned[i] or chop_aligned[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price breaks above Donchian upper band OR market becomes trending
            if close[i] > donchian_high_aligned[i] or chop_aligned[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for Donchian breakout with volume confirmation and ranging market
            bullish_breakout = (close[i] > donchian_high_aligned[i]) and volume_confirmed and ranging_market
            bearish_breakout = (close[i] < donchian_low_aligned[i]) and volume_confirmed and ranging_market
            
            if bullish_breakout:
                position = 1
                signals[i] = 0.25
            elif bearish_breakout:
                position = -1
                signals[i] = -0.25
    
    return signals