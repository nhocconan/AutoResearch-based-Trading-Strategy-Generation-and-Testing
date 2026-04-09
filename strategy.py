#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot levels from 1d + volume spike + choppiness regime filter
# Uses proven ETH/USD winning pattern: Camarilla pivots + volume + chop filter
# Adapted to 12h timeframe with 1d HTF for pivot calculation
# Discrete sizing 0.25 limits drawdown, volume confirmation reduces false breakouts
# Choppiness regime filter avoids whipsaws in ranging markets
# Works in bull/bear: mean reversion at extremes in range, breakouts in trends

name = "12h_1d_camarilla_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (based on previous day's OHLC)
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # Formula: 
    # H4 = close + 1.5*(high - low)
    # H3 = close + 1.25*(high - low)
    # H2 = close + 1.0*(high - low)
    # H1 = close + 0.5*(high - low)
    # L1 = close - 0.5*(high - low)
    # L2 = close - 1.0*(high - low)
    # L3 = close - 1.25*(high - low)
    # L4 = close - 1.5*(high - low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h4 = np.full(len(df_1d), np.nan)
    camarilla_h3 = np.full(len(df_1d), np.nan)
    camarilla_h2 = np.full(len(df_1d), np.nan)
    camarilla_h1 = np.full(len(df_1d), np.nan)
    camarilla_l1 = np.full(len(df_1d), np.nan)
    camarilla_l2 = np.full(len(df_1d), np.nan)
    camarilla_l3 = np.full(len(df_1d), np.nan)
    camarilla_l4 = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        if i < 1:  # Need previous day's data
            camarilla_h4[i] = np.nan
            camarilla_h3[i] = np.nan
            camarilla_h2[i] = np.nan
            camarilla_h1[i] = np.nan
            camarilla_l1[i] = np.nan
            camarilla_l2[i] = np.nan
            camarilla_l3[i] = np.nan
            camarilla_l4[i] = np.nan
        else:
            prev_high = high_1d[i-1]
            prev_low = low_1d[i-1]
            prev_close = close_1d[i-1]
            rang = prev_high - prev_low
            
            camarilla_h4[i] = prev_close + 1.5 * rang
            camarilla_h3[i] = prev_close + 1.25 * rang
            camarilla_h2[i] = prev_close + 1.0 * rang
            camarilla_h1[i] = prev_close + 0.5 * rang
            camarilla_l1[i] = prev_close - 0.5 * rang
            camarilla_l2[i] = prev_close - 1.0 * rang
            camarilla_l3[i] = prev_close - 1.25 * rang
            camarilla_l4[i] = prev_close - 1.5 * rang
    
    # Align 1d Camarilla levels to 12h timeframe (wait for 1d bar close)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    # Calculate Choppiness Index regime filter (14-period)
    chop = np.full(n, np.nan)
    for i in range(n):
        if i < 14:
            chop[i] = np.nan
        else:
            # True range calculation
            tr1 = high[i-14:i] - low[i-14:i]
            tr2 = np.abs(high[i-14:i] - np.roll(close[i-14:i], 1))
            tr3 = np.abs(low[i-14:i] - np.roll(close[i-14:i], 1))
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            # Skip first element of rolled arrays (no previous close)
            tr[0] = high[i-14:i][0] - low[i-14:i][0]
            atr = np.sum(tr) / 14
            
            if atr > 0:
                chop[i] = 100 * np.log10(np.sum(tr) / (atr * 14)) / np.log10(14)
            else:
                chop[i] = 50.0  # Neutral when ATR is zero
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(avg_volume[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirmed = volume[i] > 2.0 * avg_volume[i]
        
        # Choppiness regime: CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending
        is_ranging = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        
        if position == 1:  # Long position
            # Exit: price < L3 (mean reversion in range) OR price < L4 (stop in trend)
            if is_ranging and close[i] < l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif is_trending and close[i] < l4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > H3 (mean reversion in range) OR price > H4 (stop in trend)
            if is_ranging and close[i] > h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif is_trending and close[i] > h4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation and Camarilla levels
            if volume_confirmed:
                if is_ranging:
                    # In ranging market: mean reversion at H3/L3
                    if close[i] <= h3_aligned[i] and close[i] >= l3_aligned[i]:
                        # Look for rejection at extremes
                        if i >= 2:
                            # Long: rejection from H3 (price tried to go above but closed below)
                            if close[i-1] > h3_aligned[i-1] and close[i] <= h3_aligned[i]:
                                position = 1
                                signals[i] = 0.25
                            # Short: rejection from L3 (price tried to go below but closed above)
                            elif close[i-1] < l3_aligned[i-1] and close[i] >= l3_aligned[i]:
                                position = -1
                                signals[i] = -0.25
                else:  # Trending market
                    # In trending market: breakout of H4/L4 with continuation
                    if close[i] > h4_aligned[i]:
                        position = 1
                        signals[i] = 0.25
                    elif close[i] < l4_aligned[i]:
                        position = -1
                        signals[i] = -0.25
    
    return signals