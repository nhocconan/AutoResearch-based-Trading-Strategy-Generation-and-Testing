#!/usr/bin/env python3
name = "6h_WeeklyPivot_RangeReversal"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 300:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly high/low/close from daily data (last completed week)
    # Use 5 trading days of daily data (approximates a week)
    window_days = 5
    weekly_high = pd.Series(high).rolling(window=window_days*24//6, min_periods=window_days*24//6).max().values
    weekly_low = pd.Series(low).rolling(window=window_days*24//6, min_periods=window_days*24//6).min().values
    weekly_close = pd.Series(close).rolling(window=window_days*24//6, min_periods=window_days*24//6).mean().values
    
    # Weekly pivot points
    pp = (weekly_high + weekly_low + weekly_close) / 3
    r1 = 2 * pp - weekly_low
    s1 = 2 * pp - weekly_high
    r2 = pp + (weekly_high - weekly_low)
    s2 = pp - (weekly_high - weekly_low)
    
    # Align weekly pivot levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Daily trend filter: EMA(34) on daily close
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Range detection: Bollinger Bands width percentile (20-period)
    close_series = pd.Series(close)
    sma_20 = close_series.rolling(window=20, min_periods=20).mean().values
    std_20 = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    bb_width = (bb_upper - bb_lower) / sma_20
    bb_width_percentile = pd.Series(bb_width).rolling(window=100, min_periods=50).rank(pct=True).values
    
    # Volume filter: above average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20*24//6, 20, 34)  # Wait for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(pp_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(bb_width_percentile[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Range condition: low volatility environment (BB width < 30th percentile)
        is_range = bb_width_percentile[i] < 0.30
        vol_filter = volume[i] > vol_ma_20[i] * 1.2
        
        if position == 0:
            if is_range and vol_filter:
                # Long near S1 support with rejection
                if close[i] > s1_aligned[i] and close[i-1] <= s1_aligned[i-1]:
                    signals[i] = 0.25
                    position = 1
                # Short near R1 resistance with rejection
                elif close[i] < r1_aligned[i] and close[i-1] >= r1_aligned[i-1]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit: price reaches P-P or R1, or range breaks down
            if (close[i] >= pp_aligned[i] or close[i] >= r1_aligned[i] or 
                bb_width_percentile[i] > 0.70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price reaches P-P or S1, or range breaks down
            if (close[i] <= pp_aligned[i] or close[i] <= s1_aligned[i] or 
                bb_width_percentile[i] > 0.70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6s weekly pivot range reversal in low volatility environments
# - Weekly pivot S1/R1 act as strong support/resistance in ranging markets
# - Enter long at S1 bounce, short at R1 rejection during low volatility (BB width < 30th percentile)
# - Volume filter ensures institutional participation (1.2x average volume)
# - Exit at weekly pivot (PP) or opposite pivot level, or when volatility expands
# - Works in both bull/bear: ranges occur in all markets, pivot levels provide structure
# - Position size 0.25 targets 50-100 trades/year, avoiding excessive fee drag
# - Weekly pivot from daily data provides reliable structure that adapts to regimes