#!/usr/bin/env python3
# 4h_camarilla_pivot_volume_regime_v1
# Hypothesis: 4h strategy using Camarilla pivot levels from 1d timeframe for structure, volume confirmation, and choppiness regime filter.
# Long when price touches S3 level with volume > 1.3x 20-period average and chop > 61.8 (ranging market).
# Short when price touches R3 level with volume > 1.3x 20-period average and chop > 61.8 (ranging market).
# Exit when price moves back to H4/L4 levels or chop < 38.2 (trending market).
# Uses discrete position sizing (0.25) to minimize fee churn.
# Target: 20-50 trades/year (80-200 total over 4 years) on BTC/ETH/SOL to avoid overtrading and fee drag.
# Works in both bull and bear markets: Camarilla pivots provide dynamic support/resistance, volume confirms conviction at extremes,
# chop filter ensures mean reversion only in ranging markets where pivots are most effective.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pivot_volume_regime_v1"
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
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    # HLC = previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values  # Previous day's high
    prev_low = df_1d['low'].shift(1).values    # Previous day's low
    prev_close = df_1d['close'].shift(1).values # Previous day's close
    
    # Camarilla pivot levels
    # R4 = Close + ((High - Low) * 1.1/2)
    # R3 = Close + ((High - Low) * 1.1/4)
    # R2 = Close + ((High - Low) * 1.1/6)
    # R1 = Close + ((High - Low) * 1.1/12)
    # PP = (High + Low + Close)/3
    # S1 = Close - ((High - Low) * 1.1/12)
    # S2 = Close - ((High - Low) * 1.1/6)
    # S3 = Close - ((High - Low) * 1.1/4)
    # S4 = Close - ((High - Low) * 1.1/2)
    
    hl_range = prev_high - prev_low
    camarilla_r3 = prev_close + (hl_range * 1.1 / 4)
    camarilla_s3 = prev_close - (hl_range * 1.1 / 4)
    camarilla_r4 = prev_close + (hl_range * 1.1 / 2)
    camarilla_s4 = prev_close - (hl_range * 1.1 / 2)
    camarilla_h4 = prev_close + (hl_range * 1.1 / 6)  # R2 equivalent
    camarilla_l4 = prev_close - (hl_range * 1.1 / 6)  # S2 equivalent
    
    # Align HTF Camarilla levels to LTF
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
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
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(chop[i]) or
            np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or
            np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        # Regime filter: chop > 61.8 indicates ranging market (good for mean reversion)
        ranging_market = chop[i] > 61.8
        # Exit regime: chop < 38.2 indicates strong trend (exit positions)
        strong_trend = chop[i] < 38.2
        
        if position == 1:  # Long position
            # Exit conditions: price reaches H4 level OR strong trend emerges
            if high[i] >= camarilla_h4_aligned[i] or strong_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: price reaches L4 level OR strong trend emerges
            if low[i] <= camarilla_l4_aligned[i] or strong_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry conditions: price touches S3/R3 levels with volume confirmation in ranging market
            bullish_entry = (low[i] <= camarilla_s3_aligned[i]) and volume_confirmed and ranging_market
            bearish_entry = (high[i] >= camarilla_r3_aligned[i]) and volume_confirmed and ranging_market
            
            if bullish_entry:
                position = 1
                signals[i] = 0.25
            elif bearish_entry:
                position = -1
                signals[i] = -0.25
    
    return signals