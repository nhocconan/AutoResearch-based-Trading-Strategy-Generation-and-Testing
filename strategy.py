#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using Bollinger Band width for regime detection, 
# combined with price position relative to Bollinger Bands for mean reversion.
# In high volatility (BB width > 60th percentile), use breakout logic.
# In low volatility (BB width < 40th percentile), use mean reversion at band extremes.
# Uses 1-day ATR for volatility normalization and 1-day close trend filter.
# Designed to work in both bull and bear markets by adapting to volatility regimes.
# Target: 15-25 trades/year per symbol with disciplined entries.
name = "12h_BBWidth_Regime_MeanReversion_Breakout"
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
    
    # Daily ATR for volatility normalization and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate True Range components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Daily close trend (above/below previous day close)
    prev_close_1d = np.concatenate([[close_1d[0]], close_1d[:-1]])
    close_above_prev = close_1d > prev_close_1d
    close_above_prev_aligned = align_htf_to_ltf(prices, df_1d, close_above_prev.astype(float))
    
    # Bollinger Bands (20, 2) on 12h timeframe
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    sma_20_12h = pd.Series(close_12h).rolling(window=20, min_periods=20).mean().values
    std_20_12h = pd.Series(close_12h).rolling(window=20, min_periods=20).std().values
    
    upper_bb = sma_20_12h + (2 * std_20_12h)
    lower_bb = sma_20_12h - (2 * std_20_12h)
    bb_width = ((upper_bb - lower_bb) / sma_20_12h) * 100  # Percentage width
    
    # Align BB components to 12h timeframe
    sma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, sma_20_12h)
    upper_bb_aligned = align_htf_to_ltf(prices, df_12h, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_12h, lower_bb)
    bb_width_aligned = align_htf_to_ltf(prices, df_12h, bb_width)
    
    # Calculate BB width percentile rank (lookback 50 periods)
    bb_width_rank = np.full_like(bb_width_aligned, np.nan)
    for i in range(50, len(bb_width_aligned)):
        if not np.isnan(bb_width_aligned[i]):
            window = bb_width_aligned[max(0, i-49):i+1]
            valid_window = window[~np.isnan(window)]
            if len(valid_window) > 0:
                bb_width_rank[i] = (np.sum(valid_window <= bb_width_aligned[i]) / len(valid_window)) * 100
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(close_above_prev_aligned[i]) or
            np.isnan(sma_20_12h_aligned[i]) or np.isnan(upper_bb_aligned[i]) or
            np.isnan(lower_bb_aligned[i]) or np.isnan(bb_width_rank[i])):
            signals[i] = 0.0
            continue
        
        # Regime determination based on BB width
        high_volatility = bb_width_rank[i] > 60  # Volatile regime - breakout
        low_volatility = bb_width_rank[i] < 40   # Quiet regime - mean reversion
        
        if position == 0:
            # Determine trend bias from daily close vs previous close
            bullish_trend = close_above_prev_aligned[i] > 0.5
            bearish_trend = close_above_prev_aligned[i] < 0.5
            
            if high_volatility:
                # Breakout mode in volatile markets
                # Long: price breaks above upper BB with bullish daily trend
                if (close[i] > upper_bb_aligned[i] and bullish_trend):
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below lower BB with bearish daily trend
                elif (close[i] < lower_bb_aligned[i] and bearish_trend):
                    signals[i] = -0.25
                    position = -1
            elif low_volatility:
                # Mean reversion mode in quiet markets
                # Long: price at lower BB with bullish daily trend (bounce expectation)
                if (close[i] <= lower_bb_aligned[i] and bullish_trend):
                    signals[i] = 0.25
                    position = 1
                # Short: price at upper BB with bearish daily trend (fade expectation)
                elif (close[i] >= upper_bb_aligned[i] and bearish_trend):
                    signals[i] = -0.25
                    position = -1
                
        elif position == 1:
            # Long exit conditions
            if high_volatility:
                # Exit breakout: price returns to middle of BB or trend turns bearish
                if (close[i] < sma_20_12h_aligned[i]) or (not bullish_trend):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # Exit mean reversion: price reaches middle BB or stops at lower BB
                if (close[i] >= sma_20_12h_aligned[i]) or (close[i] <= lower_bb_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:
            # Short exit conditions
            if high_volatility:
                # Exit breakout: price returns to middle of BB or trend turns bullish
                if (close[i] > sma_20_12h_aligned[i]) or (not bearish_trend):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # Exit mean reversion: price reaches middle BB or stops at upper BB
                if (close[i] <= sma_20_12h_aligned[i]) or (close[i] >= upper_bb_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals