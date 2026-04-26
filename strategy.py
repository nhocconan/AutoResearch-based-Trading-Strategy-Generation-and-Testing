#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_ChopFilter_v1
Hypothesis: 12h Camarilla pivot breakout with 1d trend filter and choppiness regime to avoid whipsaws.
- Uses 12h timeframe targeting 50-150 total trades over 4 years (12-37/year)
- Camarilla R1/S1 from previous 1d bar (more stable than 12h)
- Long when price breaks above R1 with volume spike, 1d uptrend, and low chop (trending market)
- Short when price breaks below S1 with volume spike, 1d downtrend, and low chop
- Choppiness filter avoids ranging markets where breakouts fail
- Designed for low trade frequency to minimize fee drag while maintaining edge in bull/bear
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    camarilla_range = (high_1d - low_1d) * 1.1 / 12
    r1_1d = close_1d_arr + camarilla_range
    s1_1d = close_1d_arr - camarilla_range
    
    # Align Camarilla levels to 12h timeframe (wait for completed 1d bar)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Calculate volume spike (20-period volume average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)  # Volume at least 2x average
    
    # Calculate Choppiness Index on 1d to filter ranging markets
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (highest_high - lowest_low)))
    # Simplified: use rolling max/min and ATR
    tr1 = np.maximum(high_1d - low_1d, np.absolute(high_1d - np.roll(close_1d, 1)))
    tr2 = np.maximum(np.absolute(low_1d - np.roll(close_1d, 1)), tr1)
    tr1[0] = high_1d[0] - low_1d[0]  # First TR
    atr14 = pd.Series(tr2).rolling(window=14, min_periods=14).mean().values
    
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    hl_range_14 = highest_high_14 - lowest_low_14
    hl_range_14 = np.where(hl_range_14 == 0, 1e-10, hl_range_14)
    
    chop_1d = 100 * np.log10(atr14 * 14 / np.log10(14) / hl_range_14) / np.log10(100)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for volume MA, 34 for EMA, 14 for ATR)
    start_idx = max(20, 34, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(chop_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Camarilla breakout conditions with volume confirmation and regime filter
        price_above_r1 = close[i] > r1_aligned[i]
        price_below_s1 = close[i] < s1_aligned[i]
        
        # 1d trend filter
        trend_up = close[i] > ema34_1d_aligned[i]
        trend_down = close[i] < ema34_1d_aligned[i]
        
        # Choppiness filter: only trade when market is trending (CHOP < 38.2)
        trending_market = chop_aligned[i] < 38.2
        
        if position == 0:
            # Long: price breaks above R1 AND volume spike AND 1d uptrend AND trending market
            if price_above_r1 and volume_spike[i] and trend_up and trending_market:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND volume spike AND 1d downtrend AND trending market
            elif price_below_s1 and volume_spike[i] and trend_down and trending_market:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below S1 OR 1d trend turns down OR market becomes choppy
            if price_below_s1 or not trend_up or not trending_market:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above R1 OR 1d trend turns up OR market becomes choppy
            if price_above_r1 or not trend_down or not trending_market:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0