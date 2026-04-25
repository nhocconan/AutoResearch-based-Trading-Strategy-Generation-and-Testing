#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dRegime_ChopFilter_v1
Hypothesis: Trade Camarilla R1/S1 breakouts on 4h timeframe with 1-day trend filter and choppiness regime filter. 
In trending markets (ADX > 25): buy when price breaks above Camarilla R1 and price > daily EMA34. 
In ranging markets (ADX <= 25): buy when price touches Camarilla S1 and price < daily EMA34 (mean reversion). 
Requires volume > 1.3x 20-period average for confirmation. 
Exit on opposite Camarilla level touch or trend/reversal signal. 
Position size: 0.25 to limit drawdown. 
Target: 75-200 total trades over 4 years = 19-50/year. 
Works in bull (breakouts with uptrend) and bear (breakdowns with downtrend) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels, trend filter, and regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need sufficient data for EMA34 and ADX
        return np.zeros(n)
    
    # Calculate daily EMA34 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period average volume for confirmation
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate ADX for regime filter (trending vs ranging)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+
    tr_period = 14
    tr_sum = pd.Series(tr).rolling(window=tr_period, min_periods=tr_period).sum().values
    dm_plus_sum = pd.Series(dm_plus).rolling(window=tr_period, min_periods=tr_period).sum().values
    dm_minus_sum = pd.Series(dm_minus).rolling(window=tr_period, min_periods=tr_period).sum().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_sum / tr_sum
    di_minus = 100 * dm_minus_sum / tr_sum
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=tr_period, min_periods=tr_period).mean().values
    
    # Align HTF indicators
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Camarilla levels for each 1d bar
    hl_range_1d = high_1d - low_1d
    r1_1d = close_1d + (1.1 * hl_range_1d / 12)  # R1 = close + 1.1*(high-low)/12
    s1_1d = close_1d - (1.1 * hl_range_1d / 12)  # S1 = close - 1.1*(high-low)/12
    
    # Align Camarilla levels
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34 (34), volume MA (20), and ADX (14+14=28)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend (bullish = price above daily EMA34)
        htf_1d_bullish = close[i] > ema_34_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirm = volume[i] > 1.3 * vol_ma_20_aligned[i]
        
        # Regime filter: ADX > 25 = trending, ADX <= 25 = ranging
        is_trending = adx_aligned[i] > 25
        is_ranging = adx_aligned[i] <= 25
        
        if position == 0:
            # Long setup depends on regime
            if is_trending:
                # In trending market: breakout above R1 with uptrend + volume
                long_setup = (close[i] > r1_aligned[i]) and htf_1d_bullish and volume_confirm
            else:
                # In ranging market: mean reversion at S1 with downtrend + volume
                long_setup = (close[i] < s1_aligned[i]) and htf_1d_bearish and volume_confirm
            
            # Short setup depends on regime
            if is_trending:
                # In trending market: breakdown below S1 with downtrend + volume
                short_setup = (close[i] < s1_aligned[i]) and htf_1d_bearish and volume_confirm
            else:
                # In ranging market: mean reversion at R1 with uptrend + volume
                short_setup = (close[i] > r1_aligned[i]) and htf_1d_bullish and volume_confirm
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price touches opposite level (S1 for long) OR trend/reversal signal
            if (close[i] <= s1_aligned[i]) or \
               (is_trending and not htf_1d_bullish) or \
               (not is_trending and htf_1d_bullish):  # Exit ranging long when trend turns up
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches opposite level (R1 for short) OR trend/reversal signal
            if (close[i] >= r1_aligned[i]) or \
               (is_trending and not htf_1d_bearish) or \
               (not is_trending and not htf_1d_bullish):  # Exit ranging short when trend turns down or ranging long signal
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dRegime_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0