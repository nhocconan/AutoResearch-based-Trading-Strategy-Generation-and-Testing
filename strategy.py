#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_Volume_Confirmation_and_Chop_Regime
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
combined with volume confirmation (>1.5x 20-day average) and choppiness regime filter
(CHOP > 50 for ranging markets where mean reversion works, CHOP < 50 for trending markets
where trend following works). This strategy adapts to both bull and bear markets by using
regime-appropriate logic: mean reversion in choppy regimes, trend following in trending regimes.
Uses discrete sizing 0.25 to limit fee drag. Target 7-25 trades/year on 1d timeframe.
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
    volume = prices['volume'].values
    
    # Get 1w data for higher timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate KAMA on 1d (ER=10, fast=2, slow=30)
    close_s = pd.Series(close)
    change = abs(close_s - close_s.shift(10)).values
    volatility = abs(close_s.diff()).rolling(window=10, min_periods=10).sum().values
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/2 - 1/30) + 1/30) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Align 1w close to 1d (completed 1w bar only)
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    # Choppiness Index: CHOP > 50 = ranging, CHOP < 50 = trending
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    highest_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    
    chop = np.zeros_like(close)
    for i in range(atr_period, len(close)):
        if highest_high[i] != lowest_low[i]:
            chop[i] = 100 * np.log10(sum(tr[i-atr_period+1:i+1]) / np.log(atr_period) / (highest_high[i] - lowest_low[i]))
        else:
            chop[i] = 50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for KAMA (10), ATR (14), volume MA (20)
    start_idx = max(20, 14, 10)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or 
            np.isnan(close_1w_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Determine regime: choppy (>50) or trending (<50)
            is_choppy = chop[i] > 50
            
            if is_choppy:
                # In choppy regime: mean reversion
                # Long: price below KAMA + volume confirmation
                long_setup = (close[i] < kama[i]) and volume_confirm[i]
                # Short: price above KAMA + volume confirmation
                short_setup = (close[i] > kama[i]) and volume_confirm[i]
            else:
                # In trending regime: trend following with 1w filter
                # Long: price above KAMA + above 1w close + volume confirmation
                long_setup = (close[i] > kama[i]) and (close[i] > close_1w_aligned[i]) and volume_confirm[i]
                # Short: price below KAMA + below 1w close + volume confirmation
                short_setup = (close[i] < kama[i]) and (close[i] < close_1w_aligned[i]) and volume_confirm[i]
            
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
            # Exit conditions
            if is_choppy := (chop[i] > 50):
                # In choppy: exit mean reversion when price crosses KAMA
                if close[i] > kama[i]:
                    signals[i] = 0.0
                    position = 0
            else:
                # In trending: exit when price crosses below KAMA or below 1w close
                if (close[i] < kama[i]) or (close[i] < close_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions
            if is_choppy := (chop[i] > 50):
                # In choppy: exit mean reversion when price crosses KAMA
                if close[i] < kama[i]:
                    signals[i] = 0.0
                    position = 0
            else:
                # In trending: exit when price crosses above KAMA or above 1w close
                if (close[i] > kama[i]) or (close[i] > close_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
    
    return signals

name = "1d_KAMA_Trend_With_Volume_Confirmation_and_Chop_Regime"
timeframe = "1d"
leverage = 1.0