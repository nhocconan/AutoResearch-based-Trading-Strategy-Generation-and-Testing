#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and choppiness regime filter
# - Uses Camarilla pivot levels (H3/L3) from 1d for breakout entries
# - Confirms with 1d volume > 1.8x 20-period average (strong institutional participation)
# - Filters by 1d choppiness index: trade when CHOP < 50 (trending) OR CHOP > 50 AND price near mean reversion zone
# - Exits on opposite Camarilla level touch (H4/L4) or time-based exit (max 3 bars)
# - Position size: 0.28 (28% of capital) to balance return and drawdown
# - Target: 12-30 trades/year on 12h timeframe (48-120 total over 4 years) to minimize fee drag
# - Works in bull markets (breakouts continue) and bear markets (mean reversion in range)

name = "12h_1d_camarilla_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute HTF indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d True Range for choppiness
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # 1d Choppiness Index(14)
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denom = np.where((highest_14 - lowest_14) > 0, highest_14 - lowest_14, 1e-10)
    chop = 100 * np.log10(sum_tr_14 / chop_denom) / np.log10(14)
    
    # 1d Camarilla pivot levels (based on previous day)
    # H4 = close + 1.5*(high-low), H3 = close + 1.0*(high-low)
    # L3 = close - 1.0*(high-low), L4 = close - 1.5*(high-low)
    daily_range = high_1d - low_1d
    camarilla_h4 = close_1d + 1.5 * daily_range
    camarilla_h3 = close_1d + 1.0 * daily_range
    camarilla_l3 = close_1d - 1.0 * daily_range
    camarilla_l4 = close_1d - 1.5 * daily_range
    camarilla_mean = (camarilla_h3 + camarilla_l3) / 2  # midpoint
    
    # 1d Volume > 1.8x 20-period average
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.8 * avg_volume_20)
    
    # Regime filters
    chop_trending = chop < 40  # Strong trending market
    chop_ranging = chop > 60   # Strong ranging market
    chop_transition = (chop >= 40) & (chop <= 60)  # Transition zone
    
    # Align all 1d indicators to 12h
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_mean_aligned = align_htf_to_ltf(prices, df_1d, camarilla_mean)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    chop_trending_aligned = align_htf_to_ltf(prices, df_1d, chop_trending.astype(float))
    chop_ranging_aligned = align_htf_to_ltf(prices, df_1d, chop_ranging.astype(float))
    chop_transition_aligned = align_htf_to_ltf(prices, df_1d, chop_transition.astype(float))
    
    # 12h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    bars_in_trade = 0
    max_bars = 3  # Maximum 3 bars (36 hours) in trade
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(camarilla_mean_aligned[i]) or np.isnan(volume_spike_aligned[i]) or
            np.isnan(chop_trending_aligned[i]) or np.isnan(chop_ranging_aligned[i]) or
            np.isnan(chop_transition_aligned[i])):
            signals[i] = 0.0
            bars_in_trade = 0
            continue
        
        # Time-based exit
        if position != 0:
            bars_in_trade += 1
            if bars_in_trade >= max_bars:
                position = 0
                bars_in_trade = 0
                signals[i] = 0.0
                continue
        
        if position == 1:  # Long position
            # Exit conditions: touch H4 (profit target) or L3 (stop/reversal)
            if high[i] >= camarilla_h4_aligned[i]:  # Profit target
                position = 0
                bars_in_trade = 0
                signals[i] = 0.0
            elif low[i] <= camarilla_l3_aligned[i]:  # Stop/reversal
                position = 0
                bars_in_trade = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.28
                
        elif position == -1:  # Short position
            # Exit conditions: touch L4 (profit target) or H3 (stop/reversal)
            if low[i] <= camarilla_l4_aligned[i]:  # Profit target
                position = 0
                bars_in_trade = 0
                signals[i] = 0.0
            elif high[i] >= camarilla_h3_aligned[i]:  # Stop/reversal
                position = 0
                bars_in_trade = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.28
        else:  # Flat
            bars_in_trade = 0
            # Look for Camarilla breakout with volume confirmation
            if (high[i] >= camarilla_h4_aligned[i] and  # Break above H4
                volume_spike_aligned[i]):               # Volume confirmation
                position = 1
                signals[i] = 0.28
            elif (low[i] <= camarilla_l4_aligned[i] and   # Break below L4
                  volume_spike_aligned[i]):               # Volume confirmation
                position = -1
                signals[i] = -0.28
            # Mean reversion in ranging market: buy near L3, sell near H3
            elif (chop_ranging_aligned[i] and 
                  low[i] <= camarilla_l3_aligned[i] * 1.005 and  # Near L3 with small buffer
                  close[i] > camarilla_mean_aligned[i]):         # But above mean (bullish bias)
                position = 1
                signals[i] = 0.28
            elif (chop_ranging_aligned[i] and 
                  high[i] >= camarilla_h3_aligned[i] * 0.995 and # Near H3 with small buffer
                  close[i] < camarilla_mean_aligned[i]):         # But below mean (bearish bias)
                position = -1
                signals[i] = -0.28
    
    return signals