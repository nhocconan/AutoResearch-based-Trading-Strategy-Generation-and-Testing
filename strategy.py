#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_ChopRegime
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and chop regime (BW < 50th percentile).
Only trade in direction of daily trend during low-volatility (choppy) markets to avoid false breakouts.
Uses discrete position sizing (0.30) to balance return and fee drag. Target: 20-40 trades/year.
Designed to work in both bull and bear markets via trend alignment and regime filtering.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for HTF trend filter and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA34 on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Camarilla pivots from previous 1d bar
    camarilla_R1 = (close_1d + (high_1d - low_1d) * 1.1 / 12)
    camarilla_S1 = (close_1d - (high_1d - low_1d) * 1.1 / 12)
    
    # Align HTF EMA34 to 4h timeframe (standard 1-bar delay for EMA)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d, additional_delay_bars=1)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1, additional_delay_bars=1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1, additional_delay_bars=1)
    
    # Calculate Bollinger Band Width regime filter on 4h (low BBW = chop/low volatility)
    bb_period = 20
    bb_std = 2.0
    sma_bb = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_bb = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma_bb + bb_std * std_bb
    lower_bb = sma_bb - bb_std * std_bb
    bb_width = (upper_bb - lower_bb) / sma_bb
    # Regime: chop when BBW < 50th percentile (rolling lookback 100 bars)
    bbw_percentile = pd.Series(bb_width).rolling(window=100, min_periods=20).quantile(0.5).values
    chop_regime = bb_width < bbw_percentile  # True when in choppy/low volatility regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34 (34) and BBands (100 for percentile)
    start_idx = max(34, 100)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_aligned[i]) or 
            np.isnan(camarilla_R1_aligned[i]) or
            np.isnan(camarilla_S1_aligned[i]) or
            np.isnan(chop_regime[i])):
            signals[i] = 0.0 if position == 0 else (0.30 if position == 1 else -0.30)
            continue
        
        if position == 0:
            # Look for Camarilla breakout signals with trend and regime filters
            # Long: price breaks above R1 in uptrend (close > EMA34) AND chop regime
            # Short: price breaks below S1 in downtrend (close < EMA34) AND chop regime
            long_signal = (close[i] > camarilla_R1_aligned[i]) and (close[i] > ema34_aligned[i]) and chop_regime[i]
            short_signal = (close[i] < camarilla_S1_aligned[i]) and (close[i] < ema34_aligned[i]) and chop_regime[i]
            
            if long_signal:
                signals[i] = 0.30
                position = 1
            elif short_signal:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.30
            # Exit when price moves back below EMA34 (trend reversal) OR chop regime ends
            exit_signal = (close[i] < ema34_aligned[i]) or (~chop_regime[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.30
            # Exit when price moves back above EMA34 (trend reversal) OR chop regime ends
            exit_signal = (close[i] > ema34_aligned[i]) or (~chop_regime[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_ChopRegime"
timeframe = "4h"
leverage = 1.0