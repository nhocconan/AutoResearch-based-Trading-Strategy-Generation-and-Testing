#!/usr/bin/env python3
"""
4h_FVG_FairValueGap_Retest_V1
4h strategy trading Fair Value Gap (FVG) retests with 1w trend filter and volume confirmation.
- Long: Bullish FVG formed + price retests FVG low + 1w EMA50 > EMA200 + volume > 1.5x 20-period avg
- Short: Bearish FVG formed + price retests FVG high + 1w EMA50 < EMA200 + volume > 1.5x 20-period avg
- Exit: Opposite FVG formation or trend reversal
Designed for ~20-30 trades/year per symbol (80-120 total over 4 years)
FVG retests offer high-probability mean reversion within institutional order flow, working in both bull and bear markets
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    close_1w = df_1w['close'].values
    
    # 1w EMA50 and EMA200 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Detect FVG on 4h: Bullish FVG = gap between low[i-2] and high[i] when low[i] > high[i-2]
    # Bearish FVG = gap between high[i-2] and low[i] when high[i] < low[i-2]
    bullish_fvg_low = np.zeros(n)
    bullish_fvg_high = np.zeros(n)
    bearish_fvg_low = np.zeros(n)
    bearish_fvg_high = np.zeros(n)
    
    for i in range(2, n):
        # Bullish FVG: low[i] > high[i-2] creates gap between high[i-2] and low[i]
        if low[i] > high[i-2]:
            bullish_fvg_low[i] = high[i-2]   # bottom of gap
            bullish_fvg_high[i] = low[i]     # top of gap
        # Bearish FVG: high[i] < low[i-2] creates gap between low[i-2] and high[i]
        elif high[i] < low[i-2]:
            bearish_fvg_low[i] = low[i]      # bottom of gap
            bearish_fvg_high[i] = low[i-2]   # top of gap
    
    # Align FVG levels to current bar (no additional delay needed as FVG is confirmed on formation)
    bullish_fvg_low_aligned = align_htf_to_ltf(prices, pd.DataFrame({'low': low}), bullish_fvg_low)
    bullish_fvg_high_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high}), bullish_fvg_high)
    bearish_fvg_low_aligned = align_htf_to_ltf(prices, pd.DataFrame({'low': low}), bearish_fvg_low)
    bearish_fvg_high_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high}), bearish_fvg_high)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # need enough for EMA200 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(ema_200_1w_aligned[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(bullish_fvg_low_aligned[i]) or np.isnan(bullish_fvg_high_aligned[i]) or
            np.isnan(bearish_fvg_low_aligned[i]) or np.isnan(bearish_fvg_high_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        uptrend = ema_50_1w_aligned[i] > ema_200_1w_aligned[i]
        downtrend = ema_50_1w_aligned[i] < ema_200_1w_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        # FVG retest conditions
        # Bullish FVG retest: price touches or goes below FVG low then closes back above it
        bullish_retest = (low[i] <= bullish_fvg_low_aligned[i]) and (close[i] > bullish_fvg_low_aligned[i])
        # Bearish FVG retest: price touches or goes above FVG high then closes back below it
        bearish_retest = (high[i] >= bearish_fvg_high_aligned[i]) and (close[i] < bearish_fvg_high_aligned[i])
        
        # New FVG formation (for exit signals)
        new_bullish_fvg = not np.isnan(bullish_fvg_low[i]) and bullish_fvg_low[i] > 0
        new_bearish_fvg = not np.isnan(bearish_fvg_high[i]) and bearish_fvg_high[i] > 0
        
        if position == 0:
            # Long: bullish FVG retest + uptrend + volume confirmation
            if bullish_retest and uptrend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: bearish FVG retest + downtrend + volume confirmation
            elif bearish_retest and downtrend and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: bearish FVG formation, trend reversal, or bearish FVG retest
            if new_bearish_fvg or not uptrend or bearish_retest:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bullish FVG formation, trend reversal, or bullish FVG retest
            if new_bullish_fvg or not downtrend or bullish_retest:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_FVG_FairValueGap_Retest_V1"
timeframe = "4h"
leverage = 1.0