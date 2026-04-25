#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_Regime
Hypothesis: Camarilla R1/S1 breakouts with 1d EMA34 trend filter, volume confirmation (>2.0x 20-bar MA), and choppiness regime filter (CHOP > 50 = range). 
Only trade breakouts in non-choppy markets (CHOP <= 50) to avoid false breakouts in ranging conditions. 
Discrete sizing 0.25 balances profit and fee drag. Works in bull/bear: trend filter adapts to market direction, 
volume confirms breakout validity, and regime filter reduces whipsaws. Target 20-40 trades/year.
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
    
    # Get 1d data for trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar's OHLC
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    prev_close_1d = df_1d['close'].shift(1).values
    
    camarilla_range = prev_high_1d - prev_low_1d
    r1 = prev_close_1d + 1.1 * camarilla_range / 12   # R1 level
    s1 = prev_close_1d - 1.1 * camarilla_range / 12   # S1 level
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    # Choppiness regime filter (14-period)
    # CHOP = 100 * log10(sum(ATR) / (log10(n) * (highest_high - lowest_low))) / log10(n)
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.maximum(tr1, np.absolute(low - np.roll(close, 1)))
    tr = tr2
    tr[0] = high[0] - low[0]  # first bar TR
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(pd.Series(atr).rolling(window=14, min_periods=14).sum().values / 
                          (np.log10(14) * (highest_high - lowest_low))) / np.log10(14)
    # In choppy markets (CHOP > 50), avoid breakout trades
    regime_filter = chop <= 50  # Only trade when NOT choppy
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 1d EMA34 (34), volume MA (20), and CHOP (14+13=27)
    start_idx = max(34, 20, 27)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 AND 1d trend bullish (close > EMA34) AND volume confirm AND not choppy
            long_setup = (close[i] > r1_aligned[i]) and \
                         (close[i] > ema_34_1d_aligned[i]) and \
                         volume_confirm[i] and \
                         regime_filter[i]
            # Short: price breaks below S1 AND 1d trend bearish (close < EMA34) AND volume confirm AND not choppy
            short_setup = (close[i] < s1_aligned[i]) and \
                          (close[i] < ema_34_1d_aligned[i]) and \
                          volume_confirm[i] and \
                          regime_filter[i]
            
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
            # Exit: price re-enters Camarilla R1/S1 range OR 1d trend turns bearish OR chop increases
            if ((close[i] < r1_aligned[i] and close[i] > s1_aligned[i]) or 
                (close[i] < ema_34_1d_aligned[i]) or
                (not regime_filter[i])):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters Camarilla R1/S1 range OR 1d trend turns bullish OR chop increases
            if ((close[i] < r1_aligned[i] and close[i] > s1_aligned[i]) or 
                (close[i] > ema_34_1d_aligned[i]) or
                (not regime_filter[i])):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0