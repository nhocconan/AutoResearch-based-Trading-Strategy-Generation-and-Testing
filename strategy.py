#!/usr/bin/env python3
"""
1d_WeeklyDonchian20_Breakout_WeeklyTrend_ChopFilter_v1
Hypothesis: 1d Donchian(20) breakout with 1w trend filter and choppiness regime to avoid whipsaws.
- Uses 1d timeframe targeting 30-100 total trades over 4 years (7-25/year)
- Donchian upper/lower from previous 20d (more stable than intraday)
- Long when price breaks above upper band with volume spike, 1w uptrend, and low chop (trending market)
- Short when price breaks below lower band with volume spike, 1w downtrend, and low chop
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
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate Donchian levels from previous 20d bar
    # Donchian(20): upper = highest high of last 20 days, lower = lowest low of last 20 days
    high_1d = get_htf_data(prices, '1d')['high'].values
    low_1d = get_htf_data(prices, '1d')['low'].values
    
    highest_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe (wait for completed 1d bar)
    upper_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '1d'), highest_high_20)
    lower_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '1d'), lowest_low_20)
    
    # Calculate volume spike (20-period volume average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)  # Volume at least 2x average
    
    # Calculate Choppiness Index on 1d to filter ranging markets
    df_1d = get_htf_data(prices, '1d')
    high_1d_arr = df_1d['high'].values
    low_1d_arr = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # True Range calculation
    tr1 = np.maximum(high_1d_arr - low_1d_arr, np.absolute(high_1d_arr - np.roll(close_1d_arr, 1)))
    tr2 = np.maximum(np.absolute(low_1d_arr - np.roll(close_1d_arr, 1)), tr1)
    tr1[0] = high_1d_arr[0] - low_1d_arr[0]  # First TR
    atr14 = pd.Series(tr2).rolling(window=14, min_periods=14).mean().values
    
    highest_high_14 = pd.Series(high_1d_arr).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d_arr).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    hl_range_14 = highest_high_14 - lowest_low_14
    hl_range_14 = np.where(hl_range_14 == 0, 1e-10, hl_range_14)
    
    chop_1d = 100 * np.log10(atr14 * 14 / np.log10(14) / hl_range_14) / np.log10(100)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for Donchian, 34 for EMA, 14 for ATR, 20 for volume MA)
    start_idx = max(20, 34, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or
            np.isnan(chop_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Donchian breakout conditions with volume confirmation and regime filter
        price_above_upper = close[i] > upper_aligned[i]
        price_below_lower = close[i] < lower_aligned[i]
        
        # 1w trend filter
        trend_up = close[i] > ema34_1w_aligned[i]
        trend_down = close[i] < ema34_1w_aligned[i]
        
        # Choppiness filter: only trade when market is trending (CHOP < 38.2)
        trending_market = chop_aligned[i] < 38.2
        
        if position == 0:
            # Long: price breaks above upper band AND volume spike AND 1w uptrend AND trending market
            if price_above_upper and volume_spike[i] and trend_up and trending_market:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band AND volume spike AND 1w downtrend AND trending market
            elif price_below_lower and volume_spike[i] and trend_down and trending_market:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below lower band OR 1w trend turns down OR market becomes choppy
            if price_below_lower or not trend_up or not trending_market:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above upper band OR 1w trend turns up OR market becomes choppy
            if price_above_upper or not trend_down or not trending_market:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_WeeklyDonchian20_Breakout_WeeklyTrend_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0