#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_RegimeFilter_v2
Hypothesis: Trade Camarilla R1/S1 breakouts on 12h timeframe with 1-day EMA50 trend filter and choppiness regime filter.
Only trade when market is trending (CHOP < 38.2) to avoid whipsaws in ranging markets.
In bull markets: buy when price breaks above Camarilla R1 and price > daily EMA50.
In bear markets: sell when price breaks below Camarilla S1 and price < daily EMA50.
Requires choppiness filter to avoid false breakouts in choppy regimes.
Position size: 0.25 to limit drawdown and reduce fee churn.
Target: 12-37 trades/year to stay within 12h hard max of 200 total trades.
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
    
    # Get 1d data for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for EMA50 and choppiness
        return np.zeros(n)
    
    # Calculate daily EMA50 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate choppiness index on 1d timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # High-Low range for period
    hl_range = high_1d - low_1d
    sum_hl_14 = pd.Series(hl_range).rolling(window=14, min_periods=14).sum().values
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index: CHOP = 100 * log10(sum_TR_14 / sum_HL_14) / log10(14)
    chop_1d = 100 * np.log10(sum_tr_14 / sum_hl_14) / np.log10(14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate Camarilla levels for 12h timeframe (using 12h OHLC)
    # For 12h timeframe, we need to calculate Camarilla from 12h data
    # But we don't have direct 12h data, so we'll approximate using price action
    # Alternative: use the same logic but on the current timeframe's recent data
    # We'll calculate Camarilla levels based on the last completed 12h bar's range
    
    # For simplicity, we'll calculate Camarilla levels on the current timeframe
    # using a rolling window approach, but we need to ensure we use completed bars
    
    # Calculate rolling high/low/close for the current timeframe
    # We'll use a 2-bar lookback to get the previous completed bar (since each bar is 12h)
    # But to be safe with alignment, we'll calculate on 1d and align, then use for 12h context
    
    # Actually, let's calculate Camarilla levels properly for the 12h timeframe
    # We need to get 12h data - but since we don't have it directly, we'll use a different approach
    # We'll calculate the Camarilla levels based on the price action within each 12h period
    # by using the high/low/close of each 12h bar
    
    # Since we're on 12h timeframe, each prices bar IS a 12h bar
    # So we can calculate Camarilla levels directly from the prices dataframe
    
    # Calculate Camarilla levels for each bar (based on previous bar's OHLC)
    # We need to shift by 1 to use previous bar's data to avoid look-ahead
    prev_high = np.concatenate([[np.nan], high[:-1]])
    prev_low = np.concatenate([[np.nan], low[:-1]])
    prev_close = np.concatenate([[np.nan], close[:-1]])
    
    # Typical price for Camarilla calculation
    hl_range = prev_high - prev_low
    r1 = prev_close + (1.1 * hl_range / 12)
    s1 = prev_close - (1.1 * hl_range / 12)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50) and choppiness (need enough for ATR calcs)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or
            np.isnan(r1[i]) or np.isnan(s1[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Regime filter: only trade when market is trending (CHOP < 38.2)
        is_trending = chop_1d_aligned[i] < 38.2
        
        # Determine 1d HTF trend (bullish = price above daily EMA50)
        htf_1d_bullish = close[i] > ema_50_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long setup: price breaks above Camarilla R1 + 1d uptrend + trending regime
            long_setup = (close[i] > r1[i]) and htf_1d_bullish and is_trending
            
            # Short setup: price breaks below Camarilla S1 + 1d downtrend + trending regime
            short_setup = (close[i] < s1[i]) and htf_1d_bearish and is_trending
            
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
            # Exit: price touches Camarilla S1 (stop) OR 1d trend turns bearish OR regime becomes choppy
            if (close[i] <= s1[i]) or (not htf_1d_bullish) or (not is_trending):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches Camarilla R1 (stop) OR 1d trend turns bullish OR regime becomes choppy
            if (close[i] >= r1[i]) or (htf_1d_bullish) or (not is_trending):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_RegimeFilter_v2"
timeframe = "12h"
leverage = 1.0