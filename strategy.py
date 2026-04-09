#!/usr/bin/env python3
# 12h_daily_camarilla_pivot_volume_regime_v2
# Hypothesis: 12h strategy using daily Camarilla pivot levels with volume confirmation and choppiness regime filter.
# Long: Price breaks above daily H3 level with volume > 2.0x 20-period average and chop > 61.8 (range market).
# Short: Price breaks below daily L3 level with volume > 2.0x 20-period average and chop > 61.8 (range market).
# Exit: Price returns to opposite pivot level (long exits below H3, short exits above L3).
# Uses 1d trend filter: only long when 1d close > 1d EMA50, only short when 1d close < 1d EMA50.
# Target: 12-30 trades/year to minimize fee drag while maintaining edge.
# Camarilla pivots provide precise support/resistance levels that work in ranging markets.
# Volume confirmation ensures institutional participation.
# Choppiness filter avoids trending markets where pivot reversals fail.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_daily_camarilla_pivot_volume_regime_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_prices = prices['open'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate previous day's Camarilla pivot levels
    # Using prior day's high, low, close (standard Camarilla calculation)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # H3 = Close + 1.1*(High-Low)/2
    # L3 = Close - 1.1*(High-Low)/2
    camarilla_h3 = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) / 2.0
    camarilla_l3 = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) / 2.0
    
    # Align Camarilla levels to 12h
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # 1d EMA50 for trend filter
    close_1d_s = pd.Series(close_1d)
    ema_50_1d = close_1d_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Choppiness Index regime filter (14-period)
    # CHOP > 61.8 = ranging market (good for mean reversion at pivots)
    # CHOP < 38.2 = trending market (avoid for pivot reversals)
    true_range = np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))
    # Handle first bar TR
    true_range[0] = high[0] - low[0]
    atr_14 = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(TR14)/(HH14-LL14)) / log10(14)
    sum_tr_14 = pd.Series(true_range).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr_14 / (highest_high_14 - lowest_low_14)) / np.log10(14)
    # Handle division by zero or invalid values
    chop = np.where((highest_high_14 - lowest_low_14) > 0, chop, 50.0)
    chop = np.where(np.isnan(chop), 50.0, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or
            np.isnan(volume[i]) or np.isnan(open_prices[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirmed = volume[i] > 2.0 * volume_ma[i]
        # Choppiness filter: chop > 61.8 = ranging market
        chop_ranging = chop[i] > 61.8
        # 1d trend filter: close > EMA50 for uptrend, < EMA50 for downtrend
        trend_1d_up = close_1d_s.iloc[i] > ema_50_1d[i] if hasattr(close_1d_s, 'iloc') else close_1d[i] > ema_50_1d[i]
        trend_1d_down = close_1d_s.iloc[i] < ema_50_1d[i] if hasattr(close_1d_s, 'iloc') else close_1d[i] < ema_50_1d[i]
        
        if position == 1:  # Long position
            # Exit: Price returns to Camarilla H3 level
            if close[i] <= camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to Camarilla L3 level
            if close[i] >= camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above Camarilla H3 with volume, chop ranging, and uptrend
            if (close[i] > camarilla_h3_aligned[i] and    # Break above H3
                volume_confirmed and                       # Volume spike
                chop_ranging and                           # Ranging market
                trend_1d_up):                              # 1d uptrend
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Camarilla L3 with volume, chop ranging, and downtrend
            elif (close[i] < camarilla_l3_aligned[i] and   # Break below L3
                  volume_confirmed and                     # Volume spike
                  chop_ranging and                         # Ranging market
                  trend_1d_down):                          # 1d downtrend
                position = -1
                signals[i] = -0.25
    
    return signals