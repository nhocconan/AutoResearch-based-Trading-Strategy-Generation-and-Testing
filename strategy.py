#!/usr/bin/env python3
# 4h_camarilla_pivot_volume_regime_v2
# Hypothesis: 4h strategy using Camarilla pivot levels from 1d HTF for mean-reversion entries. Enters long at L3 support and short at H3 resistance when volume > 1.5x 20-bar average and chop < 61.8 (trending regime). Uses discrete position sizing (0.25) and exits at L4/H4 levels or opposite pivot touch. Designed for low trade frequency (<50/year) to minimize fee drag while capturing institutional reaction at key levels. Works in bull/bear: mean reversion at pivots effective in ranging markets, volume/conviction filter avoids false signals.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pivot_volume_regime_v2"
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
    
    # Choppiness Index regime filter (14-period)
    atr_period = 14
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr_series = pd.Series(tr)
    atr_series = tr_series.rolling(window=atr_period, min_periods=atr_period).mean()
    highest_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    atr_sum = tr_series.rolling(window=atr_period, min_periods=atr_period).sum().values
    denominator = np.log10(atr_period) * (highest_high - lowest_low)
    denominator = np.where(denominator == 0, np.nan, denominator)
    chop = 100 * np.log10(atr_sum / denominator)
    
    # Multi-timeframe: 1d Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    camarilla_h4 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 4
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 4
    camarilla_l4 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align Camarilla levels to 4h timeframe (completed 1d bar only)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(chop[i]) or
            np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(camarilla_l4_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        # Regime filter: chop < 61.8 indicates trending market (favorable for mean reversion)
        trending_market = chop[i] < 61.8
        
        if position == 1:  # Long position
            # Exit: price reaches L4 (take profit) or touches H3 (stop/reversal)
            if close[i] >= camarilla_l4_aligned[i] or close[i] <= camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches H4 (take profit) or touches L3 (stop/reversal)
            if close[i] <= camarilla_h4_aligned[i] or close[i] >= camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for mean reversion entry at L3/H3 with volume and regime confirmation
            long_entry = (close[i] <= camarilla_l3_aligned[i]) and volume_confirmed and trending_market
            short_entry = (close[i] >= camarilla_h3_aligned[i]) and volume_confirmed and trending_market
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals