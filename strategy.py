#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_ChopFilter
Hypothesis: Uses 12h timeframe with Camarilla R1/S1 breakouts filtered by 1d trend (price > SMA50 for long, price < SMA50 for short), volume confirmation (>2x 20-period average), and choppiness regime (CHOP > 50 for mean-reversion filter). Designed for BTC/ETH to work in both bull and bear markets by taking breakouts in trending regimes and mean-reversion in choppy markets. Target ~15-25 trades/year to minimize fee drag.
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
    
    # Get 1d data for HTF filters
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA50 trend filter
    sma_50 = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_50_aligned = align_htf_to_ltf(prices, df_1d, sma_50)
    
    # 1d data for Camarilla levels (from previous completed 1d bar)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    rng = prev_high - prev_low
    r1 = prev_close + (rng * 1.1 / 12)
    s1 = prev_close - (rng * 1.1 / 12)
    
    # Align Camarilla levels to 12h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    # Choppiness Index regime filter (using 12h data)
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low)) / log10(14)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum) / np.log10(14) / np.log10((hh - ll) + 1e-10)
    chop_regime = chop > 50  # choppy market (>50) = mean reversion, trending (<50) = trend follow
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # fixed size to minimize churn
    
    # Warmup: need 1d SMA50 (50), 1d shift(1) for Camarilla, vol avg (20), ATR for CHOP (14)
    start_idx = max(50 + 1, 1 + 1, 20, 14)  # ~51 bars
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(sma_50_aligned[i]) or np.isnan(volume_confirm[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        sma_val = sma_50_aligned[i]
        vol_conf = volume_confirm[i]
        chop_val = chop[i]
        
        if position == 0:
            # In choppy market (CHOP > 50): mean reversion at Camarilla levels
            # In trending market (CHOP <= 50): breakout with trend filter
            if chop_val > 50:  # choppy/mean reversion regime
                long_condition = (close_val < s1_val and  # price below S1 = oversold
                                vol_conf)
                short_condition = (close_val > r1_val and  # price above R1 = overbought
                                 vol_conf)
            else:  # trending regime
                long_condition = (close_val > r1_val and   # breakout above R1
                                close_val > sma_val and    # price above 1d SMA50 = uptrend
                                vol_conf)
                short_condition = (close_val < s1_val and  # breakdown below S1
                                 close_val < sma_val and   # price below 1d SMA50 = downtrend
                                 vol_conf)
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price crosses below S1 (mean reversion) or below SMA50 (trend fail)
            if chop_val > 50:  # choppy: exit at mean reversion target
                if close_val > s1_val:  # price back above S1 = exit long
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = size
            else:  # trending: exit if trend fails
                if close_val < sma_val:  # price below SMA50 = trend failure
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = size
        elif position == -1:
            # Exit short: price crosses above R1 (mean reversion) or above SMA50 (trend fail)
            if chop_val > 50:  # choppy: exit at mean reversion target
                if close_val < r1_val:  # price back below R1 = exit short
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -size
            else:  # trending: exit if trend fails
                if close_val > sma_val:  # price above SMA50 = trend failure
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -size
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0