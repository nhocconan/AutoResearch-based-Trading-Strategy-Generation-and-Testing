#!/usr/bin/env python3
"""
12h_1d_Camarilla_Breakout_Volume_Regime_v1
Hypothesis: Trade Camarilla pivot breakouts (H4/L4 levels) on 12h chart with volume confirmation and 1-day Choppiness Index regime filter. 
In trending markets (CHOP < 38.2), trade breakouts in direction of trend. In ranging markets (CHOP > 61.8), trade mean reversion at extreme levels (H5/L5).
Designed for 15-25 trades/year with clear rules that work in both bull (breakouts continue) and bear (mean reversion in ranges) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Camarilla_Breakout_Volume_Regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY DATA FOR CAMARILLA PIVOTS AND CHOPPINESS ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day (based on previous day)
    camarilla_h5 = np.full_like(close_1d, np.nan)
    laughter_h4 = np.full_like(close_1d, np.nan)
    camarilla_l4 = np.full_like(close_1d, np.nan)
    camarilla_l5 = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        # Previous day's range
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_val = prev_high - prev_low
        
        if range_val > 0:
            camarilla_h5[i] = prev_close + 1.1 * range_val * 1.1
            laughter_h4[i] = prev_close + 1.1 * range_val * 0.5
            camarilla_l4[i] = prev_close - 1.1 * range_val * 0.5
            camarilla_l5[i] = prev_close - 1.1 * range_val * 1.1
        else:
            camarilla_h5[i] = prev_close
            laughter_h4[i] = prev_close
            camarilla_l4[i] = prev_close
            camarilla_l5[i] = prev_close
    
    # Calculate Choppiness Index (14-period)
    chop_period = 14
    atr_1d = np.zeros_like(high_1d)
    for i in range(1, len(high_1d)):
        atr_1d[i] = max(high_1d[i] - low_1d[i], 
                       abs(high_1d[i] - close_1d[i-1]), 
                       abs(low_1d[i] - close_1d[i-1]))
    
    # True Range sum and highest/lowest over period
    tr_sum = np.zeros_like(high_1d)
    highest_high = np.zeros_like(high_1d)
    lowest_low = np.zeros_like(high_1d)
    
    for i in range(len(high_1d)):
        if i == 0:
            tr_sum[i] = atr_1d[i]
            highest_high[i] = high_1d[i]
            lowest_low[i] = low_1d[i]
        else:
            tr_sum[i] = tr_sum[i-1] + atr_1d[i]
            if i >= chop_period:
                tr_sum[i] = tr_sum[i-1] - atr_1d[i-chop_period] + atr_1d[i]
            highest_high[i] = max(high_1d[i], highest_high[i-1] if i>0 else high_1d[i])
            lowest_low[i] = min(low_1d[i], lowest_low[i-1] if i>0 else low_1d[i])
            if i >= chop_period:
                highest_high[i] = max(high_1d[i-chop_period+1:i+1]) if i >= chop_period-1 else highest_high[i]
                lowest_low[i] = min(low_1d[i-chop_period+1:i+1]) if i >= chop_period-1 else lowest_low[i]
    
    # Choppiness Index formula
    chop = np.full_like(high_1d, 50.0)  # default neutral
    for i in range(chop_period-1, len(high_1d)):
        if highest_high[i] > lowest_low[i] and tr_sum[i] > 0:
            chop[i] = 100 * np.log10(tr_sum[i] / (highest_high[i] - lowest_low[i])) / np.log10(chop_period)
    
    # Align Camarilla levels and Chop to 12h timeframe
    h5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h5)
    h4_aligned = align_htf_to_ltf(prices, df_1d, laughter_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    l5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l5)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === 12H INDICATORS: VOLUME FILTER ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(h5_aligned[i]) or np.isnan(h4_aligned[i]) or 
            np.isnan(l4_aligned[i]) or np.isnan(l5_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume strength
        strong_volume = volume[i] > (vol_ma[i] * 1.5)
        
        # Regime filters
        trending_market = chop_aligned[i] < 38.2  # Trending: chop < 38.2
        ranging_market = chop_aligned[i] > 61.8   # Ranging: chop > 61.8
        
        # Initialize signal
        signal_generated = False
        
        if trending_market and strong_volume:
            # In trending market: trade breakouts in direction of trend
            # Use price action to determine trend (price vs 20-period EMA)
            if i >= 20:
                ema20 = pd.Series(close[:i+1]).ewm(span=20, adjust=False).mean().iloc[-1]
                uptrend = close[i] > ema20
                downtrend = close[i] < ema20
                
                # Long breakout above H4 in uptrend
                if uptrend and close[i] > h4_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                    signal_generated = True
                # Short breakdown below L4 in downtrend
                elif downtrend and close[i] < l4_aligned[i]:
                    position = -1
                    signals[i] = -0.25
                    signal_generated = True
                    
        elif ranging_market and strong_volume:
            # In ranging market: mean reversion at extreme levels
            # Long near L5 (strong support)
            if close[i] <= l5_aligned[i] * 1.001:  # Allow small slippage
                position = 1
                signals[i] = 0.25
                signal_generated = True
            # Short near H5 (strong resistance)
            elif close[i] >= h5_aligned[i] * 0.999:  # Allow small slippage
                position = -1
                signals[i] = -0.25
                signal_generated = True
        
        # Exit conditions (apply regardless of regime)
        if not signal_generated:
            if position == 1:
                # Exit long: price reaches H4 (take profit) or L4 (stop loss)
                if close[i] >= h4_aligned[i] or close[i] <= l4_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price reaches L4 (take profit) or H4 (stop loss)
                if close[i] <= l4_aligned[i] or close[i] >= h4_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals