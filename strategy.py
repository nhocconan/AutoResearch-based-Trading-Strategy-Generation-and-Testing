#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_VolumeSpike_ChopFilter
Hypothesis: 12h Donchian(20) breakout with volume confirmation (>2.0x 20-period MA) and choppy market filter (CHOP > 61.8 = range -> mean reversion). 
In choppy markets (CHOP > 61.8): fade the breakout (short upper band, long lower band). 
In trending markets (CHOP <= 61.8): follow the breakout (long upper band, short lower band).
Uses 1d EMA50 as additional trend filter to avoid counter-trend trades in strong trends.
Designed for 12-37 trades/year on 12h timeframe. Works in both bull and bear markets by adapting to regime.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) on 12h data
    # Use rolling window with min_periods
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_roll
    donchian_lower = low_roll
    
    # 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    uptrend_1d = close > ema_50_1d_aligned
    downtrend_1d = close < ema_50_1d_aligned
    
    # Volume confirmation: volume > 2.0x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Choppiness Index (CHOP) - calculates if market is choppy (range) or trending
    # CHOP = 100 * log10(sum(ATR(1)) / (n * (HHV - LLV))) / log10(n)
    # CHOP > 61.8 = ranging/choppy market (favor mean reversion)
    # CHOP < 38.2 = trending market (favor trend following)
    # We'll use CHOP > 61.8 as choppy filter
    tr = np.maximum(high - low, np.absolute(high - np.roll(close, 1)), np.absolute(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]  # first period
    atr1 = pd.Series(tr).rolling(window=1, min_periods=1).sum().values  # ATR(1) = TR
    atr_sum = pd.Series(tr).rolling(window=20, min_periods=20).sum().values  # sum of TR over 20 periods
    hhvl = pd.Series(high).rolling(window=20, min_periods=20).max().values
    llvl = pd.Series(low).rolling(window=20, min_periods=20).min().values
    chop_raw = 100 * np.log10(atr_sum / (20 * (hhvl - llvl))) / np.log10(20)
    choppy_market = chop_raw > 61.8  # choppy/ranging market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for Donchian/volume/CHOP, 50 for EMA)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(choppy_market[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            if choppy_market[i]:
                # Choppy market: mean reversion - fade the breakout
                # Short when price breaks above upper band (expect reversal down)
                # Long when price breaks below lower band (expect reversal up)
                if (close[i] > donchian_upper[i] and volume_spike[i]):
                    signals[i] = -0.25
                    position = -1
                elif (close[i] < donchian_lower[i] and volume_spike[i]):
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            else:
                # Trending market: follow the breakout
                # Long when price breaks above upper band with volume
                # Short when price breaks below lower band with volume
                if (close[i] > donchian_upper[i] and volume_spike[i] and uptrend_1d[i]):
                    signals[i] = 0.25
                    position = 1
                elif (close[i] < donchian_lower[i] and volume_spike[i] and downtrend_1d[i]):
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit conditions
            if choppy_market[i]:
                # In choppy market: exit long when price reaches middle of channel or breaks lower band
                middle = (donchian_upper[i] + donchian_lower[i]) / 2
                if close[i] < middle or close[i] < donchian_lower[i]:
                    signals[i] = 0.0
                    position = 0
            else:
                # In trending market: exit when price closes below lower band or trend changes
                if close[i] < donchian_lower[i] or not uptrend_1d[i]:
                    signals[i] = 0.0
                    position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit conditions
            if choppy_market[i]:
                # In choppy market: exit short when price reaches middle of channel or breaks upper band
                middle = (donchian_upper[i] + donchian_lower[i]) / 2
                if close[i] > middle or close[i] > donchian_upper[i]:
                    signals[i] = 0.0
                    position = 0
            else:
                # In trending market: exit when price closes above upper band or trend changes
                if close[i] > donchian_upper[i] or not downtrend_1d[i]:
                    signals[i] = 0.0
                    position = 0
    
    return signals

name = "12h_Donchian20_Breakout_VolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0