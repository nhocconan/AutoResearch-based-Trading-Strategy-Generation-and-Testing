#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_RegimeFilter_v1
Hypothesis: Trade Camarilla R1/S1 breakouts on 12h timeframe with 1d EMA50 trend filter and choppiness regime filter.
In trending markets (CHOP < 38.2): buy when price breaks above R1 + EMA50 uptrend, sell when breaks below S1 + EMA50 downtrend.
In ranging markets (CHOP > 61.8): fade moves at H3/L3 levels for mean reversion.
Position size: 0.25 to balance risk and reward.
Target: 12-30 trades/year to stay within 12h hard max of 200 total trades.
Uses 1d HTF for trend and regime filters to avoid look-ahead bias.
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
    
    # Get 1d data for HTF trend and regime filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for indicators
        return np.zeros(n)
    
    # Calculate 1d EMA50 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d choppiness index for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with close_1d index
    
    # ATR(14)
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of True Range over 14 periods
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index: CHOP = 100 * log10(sum_tr_14 / (ATR(14) * 14)) / log10(14)
    chop_1d = 100 * np.log10(sum_tr_14 / (atr_1d * 14)) / np.log10(14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 1d OHLC for Camarilla pivot levels (based on previous day)
    # Camarilla levels use previous day's OHLC
    prev_close_1d = np.concatenate([[np.nan], close_1d[:-1]])
    prev_open_1d = np.concatenate([[np.nan], df_1d['open'].values[:-1]])
    prev_high_1d = np.concatenate([[np.nan], df_1d['high'].values[:-1]])
    prev_low_1d = np.concatenate([[np.nan], df_1d['low'].values[:-1]])
    
    # Camarilla R1, S1, H3, L3 levels
    # R1 = Close + (High - Low) * 1.1 / 12
    # S1 = Close - (High - Low) * 1.1 / 12
    # H3 = Close + (High - Low) * 1.1 / 6
    # L3 = Close - (High - Low) * 1.1 / 6
    high_low_diff_1d = prev_high_1d - prev_low_1d
    r1_1d = prev_close_1d + high_low_diff_1d * 1.1 / 12
    s1_1d = prev_close_1d - high_low_diff_1d * 1.1 / 12
    h3_1d = prev_close_1d + high_low_diff_1d * 1.1 / 6
    l3_1d = prev_close_1d - high_low_diff_1d * 1.1 / 6
    
    # Align Camarilla levels to 12h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50) and chop (14)
    start_idx = max(50, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or
            np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend (bullish = price above EMA50)
        htf_1d_bullish = close[i] > ema_50_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_50_1d_aligned[i]
        
        # Regime filter: chop < 38.2 = trending, chop > 61.8 = ranging
        chop_value = chop_1d_aligned[i]
        is_trending = chop_value < 38.2
        is_ranging = chop_value > 61.8
        
        if position == 0:
            if is_trending:
                # Trending regime: breakout strategy
                # Long setup: price breaks above R1 + 1d uptrend
                long_setup = (close[i] > r1_1d_aligned[i]) and htf_1d_bullish
                
                # Short setup: price breaks below S1 + 1d downtrend
                short_setup = (close[i] < s1_1d_aligned[i]) and htf_1d_bearish
                
                if long_setup:
                    signals[i] = 0.25
                    position = 1
                elif short_setup:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif is_ranging:
                # Ranging regime: mean reversion at H3/L3
                # Long setup: price touches L3 (support) and shows rejection
                long_setup = (low[i] <= l3_1d_aligned[i]) and (close[i] > open_prices[i]) and htf_1d_bullish
                
                # Short setup: price touches H3 (resistance) and shows rejection
                short_setup = (high[i] >= h3_1d_aligned[i]) and (close[i] < open_prices[i]) and htf_1d_bearish
                
                if long_setup:
                    signals[i] = 0.25
                    position = 1
                elif short_setup:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # Choppy regime (38.2 <= CHOP <= 61.8): no trade
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions
            if is_trending:
                # Trending: exit on trend reversal or touch of S1 (stop)
                if (not htf_1d_bullish) or (close[i] < s1_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
            elif is_ranging:
                # Ranging: exit at midpoint or opposite level
                midpoint_1d = (h3_1d_aligned[i] + l3_1d_aligned[i]) / 2
                if close[i] >= midpoint_1d:
                    signals[i] = 0.0
                    position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions
            if is_trending:
                # Trending: exit on trend reversal or touch of R1 (stop)
                if htf_1d_bullish or (close[i] > r1_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
            elif is_ranging:
                # Ranging: exit at midpoint or opposite level
                midpoint_1d = (h3_1d_aligned[i] + l3_1d_aligned[i]) / 2
                if close[i] <= midpoint_1d:
                    signals[i] = 0.0
                    position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_RegimeFilter_v1"
timeframe = "12h"
leverage = 1.0