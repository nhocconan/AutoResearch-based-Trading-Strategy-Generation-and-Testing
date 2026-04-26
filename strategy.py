#!/usr/bin/env python3
"""
1d_WeeklyDonchian20_Breakout_WeeklyTrend_ChopFilter_v1
Hypothesis: Daily Donchian(20) breakout with weekly trend filter and choppiness regime to avoid whipsaws.
- Uses 1d timeframe targeting 30-100 total trades over 4 years (7-25/year)
- Donchian breakout from previous 20 daily bars (structure-based entry)
- Long when price breaks above 20d high with weekly uptrend and low chop (trending market)
- Short when price breaks below 20d low with weekly downtrend and low chop
- Choppiness filter (1d) avoids ranging markets where breakouts fail
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
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate Donchian channels from previous 20 daily bars
    # We need daily data for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 20-period Donchian high/low (using previous bar to avoid look-ahead)
    donch_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().shift(1).values
    donch_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align Donchian levels to 1d timeframe (already aligned via shift)
    donch_high_aligned = donch_high_20
    donch_low_aligned = donch_low_20
    
    # Calculate volume spike (20-period volume average on 1d)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)  # Volume at least 2x average
    
    # Calculate Choppiness Index on 1d to filter ranging markets
    # True Range calculation
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.maximum(np.absolute(low - np.roll(close, 1)), tr1)
    tr1[0] = high[0] - low[0]  # First TR
    tr = tr2
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    hl_range_14 = highest_high_14 - lowest_low_14
    hl_range_14 = np.where(hl_range_14 == 0, 1e-10, hl_range_14)
    
    chop_1d = 100 * np.log10(atr14 * 14 / np.log10(14) / hl_range_14) / np.log10(100)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for Donchian, 34 for EMA, 14 for ATR/CHOP)
    start_idx = max(20, 34, 14) + 1  # +1 for Donchian shift
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or
            np.isnan(chop_1d[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Donchian breakout conditions with volume confirmation and regime filter
        price_above_donch_high = close[i] > donch_high_aligned[i]
        price_below_donch_low = close[i] < donch_low_aligned[i]
        
        # Weekly trend filter
        trend_up = close[i] > ema34_1w_aligned[i]
        trend_down = close[i] < ema34_1w_aligned[i]
        
        # Choppiness filter: only trade when market is trending (CHOP < 38.2)
        trending_market = chop_1d[i] < 38.2
        
        if position == 0:
            # Long: price breaks above Donchian high AND volume spike AND weekly uptrend AND trending market
            if price_above_donch_high and volume_spike[i] and trend_up and trending_market:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND volume spike AND weekly downtrend AND trending market
            elif price_below_donch_low and volume_spike[i] and trend_down and trending_market:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below Donchian low OR weekly trend turns down OR market becomes choppy
            if price_below_donch_low or not trend_up or not trending_market:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above Donchian high OR weekly trend turns up OR market becomes choppy
            if price_above_donch_high or not trend_down or not trending_market:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_WeeklyDonchian20_Breakout_WeeklyTrend_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0