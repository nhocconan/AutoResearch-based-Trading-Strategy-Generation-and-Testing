#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1w Camarilla pivot levels with volume confirmation
# Weekly Camarilla levels provide strong support/resistance that price reacts to across market regimes
# Volume confirmation (current 12h volume > 2.0x 20-period average) filters false breakouts
# Only trade in direction of weekly EMA(20) trend to avoid counter-trend whipsaws
# Designed for 12h timeframe targeting 12-30 trades/year (48-120 over 4 years)
# Works in bull/bear: price reacts to weekly structure, volume confirms validity, EMA filter ensures trend alignment

name = "12h_1w_camarilla_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 25:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla levels (based on prior week)
    # Camarilla uses previous period's high, low, close
    # H4 = close + 1.1*(high-low)/2
    # L4 = close - 1.1*(high-low)/2
    # H3 = close + 1.1*(high-low)/4
    # L3 = close - 1.1*(high-low)/4
    # H2 = close + 1.1*(high-low)/6
    # L2 = close - 1.1*(high-low)/6
    # H1 = close + 1.1*(high-low)/12
    # L1 = close - 1.1*(high-low)/12
    
    # Shift by 1 to use prior week's data (no look-ahead)
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close = np.roll(close_1w, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla levels for prior week
    hl_range = prev_high - prev_low
    camarilla_h4 = prev_close + 1.1 * hl_range / 2
    camarilla_l4 = prev_close - 1.1 * hl_range / 2
    camarilla_h3 = prev_close + 1.1 * hl_range / 4
    camarilla_l3 = prev_close - 1.1 * hl_range / 4
    camarilla_h2 = prev_close + 1.1 * hl_range / 6
    camarilla_l2 = prev_close - 1.1 * hl_range / 6
    camarilla_h1 = prev_close + 1.1 * hl_range / 12
    camarilla_l1 = prev_close - 1.1 * hl_range / 12
    
    # Align weekly Camarilla levels to 12h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    h2_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h2)
    l2_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l2)
    h1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h1)
    l1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l1)
    
    # Calculate weekly EMA(20) for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Pre-compute volume confirmation (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(ema_20_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 2.0x average 12h volume
        volume_confirmed = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit if price drops below L3 level
            if close[i] < l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if price rises above H3 level
            if close[i] > h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Only trade with volume confirmation and trend alignment
            if volume_confirmed:
                # Long if price breaks above H4 and above weekly EMA(20)
                if close[i] > h4_aligned[i] and close[i] > ema_20_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short if price breaks below L4 and below weekly EMA(20)
                elif close[i] < l4_aligned[i] and close[i] < ema_20_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals