#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_ChopFilter_v3
Hypothesis: 12h Camarilla R1/S1 breakout with 1d trend alignment, volume spike confirmation, and choppiness regime filter. Enters long when price breaks above R1 in bullish 1d trend with volume expansion in low-chop environment; enters short when price breaks below S1 in bearish 1d trend with volume expansion. Uses discrete position sizing (0.25) to minimize fee drag. Designed for low trade frequency (12-37/year) to overcome fee drag in ranging/bear markets like 2025.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF trend, volume average, and chop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h Camarilla pivot levels (based on prior 12h bar's OHLC)
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    # We need prior bar's OHLC, so we shift by 1
    prior_high = np.roll(high, 1)
    prior_low = np.roll(low, 1)
    prior_close = np.roll(close, 1)
    prior_high[0] = high[0]  # avoid NaN on first bar
    prior_low[0] = low[0]
    prior_close[0] = close[0]
    
    camarilla_range = prior_high - prior_low
    R1 = prior_close + 1.1 * camarilla_range / 12
    S1 = prior_close - 1.1 * camarilla_range / 12
    
    # Calculate ATR for stoploss (using 12-period ATR)
    atr_period = 12
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum.reduce([tr1, tr2, tr3])
    atr = pd.Series(tr).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    
    # Calculate 1d EMA34 for HTF trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    htf_trend = np.where(close > ema_34_1d_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Calculate 1d average volume for volume spike filter
    avg_volume_1d = pd.Series(df_1d['volume'].values).ewm(span=20, min_periods=20, adjust=False).mean().values
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    volume_spike = volume > (1.5 * avg_volume_1d_aligned)  # 50% above average
    
    # Calculate choppiness index (14-period) for regime filter
    chop_period = 14
    sum_tr = pd.Series(tr).rolling(window=chop_period, min_periods=chop_period).sum().values
    highest_high = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    lowest_low = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    chop_denom = np.log10(sum_tr) * (chop_period / np.log10(2))
    chop_num = np.log10((highest_high - lowest_low) + 1e-10)
    chop = 100 * (chop_num / chop_denom)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA, 14 for chop, 12 for ATR)
    start_idx = max(34, chop_period, atr_period) + 1  # +1 for prior bar shift
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr[i]) or np.isnan(chop[i]) or np.isnan(R1[i]) or np.isnan(S1[i]) or
            np.isnan(htf_trend[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Regime filter: only trade in low-chop environment (trending market)
        # Chop < 38.2 = strong trending (ideal for breakouts)
        if chop[i] >= 38.2:
            # High chop = ranging market, avoid breakout trades
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Breakout logic with volume confirmation and HTF trend alignment
        long_entry = (close[i] > R1[i]) and volume_spike[i] and (htf_trend[i] == 1)
        short_entry = (close[i] < S1[i]) and volume_spike[i] and (htf_trend[i] == -1)
        
        # Exit logic: reverse signal or stoploss
        long_exit = (position == 1) and (close[i] < S1[i])  # reverse to short or flat
        short_exit = (position == -1) and (close[i] > R1[i])  # reverse to long or flat
        
        # ATR-based stoploss (optional - using close-based exit for simplicity)
        # In practice, engine will handle exit via signal change
        
        if long_entry and position != 1:
            signals[i] = 0.25
            position = 1
        elif short_entry and position != -1:
            signals[i] = -0.25
            position = -1
        elif long_exit:
            signals[i] = -0.25  # reverse to short
            position = -1
        elif short_exit:
            signals[i] = 0.25   # reverse to long
            position = 1
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_ChopFilter_v3"
timeframe = "12h"
leverage = 1.0