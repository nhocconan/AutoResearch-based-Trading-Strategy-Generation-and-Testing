#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrend_Regime
Hypothesis: Trade 6h Elder Ray Bull/Bear Power with 1d EMA50 trend filter and Bollinger Bandwidth regime filter. 
Long when Bull Power > 0 (close > EMA13) AND price above 1d EMA50 AND low volatility regime (BBW < 50th percentile).
Short when Bear Power < 0 (close < EMA13) AND price below 1d EMA50 AND low volatility regime.
Uses discrete sizing 0.25 to balance return and drawdown. Target 12-30 trades/year on 6h timeframe.
Elder Ray measures bull/bear strength relative to EMA13, effective in both bull and bear markets when combined with trend and regime filters.
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
    
    # Calculate EMA13 for Elder Ray (primary timeframe)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = close - ema13  # Bull Power = Close - EMA13
    bear_power = close - ema13  # Bear Power = Close - EMA13 (negative when close < EMA13)
    
    # Get 1d data for HTF trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA50 on 1d for HTF trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Bollinger Bandwidth on 1d for regime filter
    sma_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma_20_1d + (2.0 * std_20_1d)
    lower_bb_1d = sma_20_1d - (2.0 * std_20_1d)
    bb_width_1d = (upper_bb_1d - lower_bb_1d) / sma_20_1d
    
    # Calculate 50-period percentile rank of BBW for regime detection (low volatility = BBW < 50th percentile)
    bb_width_series = pd.Series(bb_width_1d)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    low_volatility_regime = bb_width_percentile < 0.5  # Low volatility regime
    low_volatility_regime_aligned = align_htf_to_ltf(prices, df_1d, low_volatility_regime)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA13 (13), 1d EMA50 (50), 1d BBW percentile (50)
    start_idx = max(13, 50, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(low_volatility_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 (close > EMA13) + above 1d EMA50 + low volatility regime
            long_setup = (bull_power[i] > 0) and \
                         (close[i] > ema_50_1d_aligned[i]) and \
                         low_volatility_regime_aligned[i]
            # Short: Bear Power < 0 (close < EMA13) + below 1d EMA50 + low volatility regime
            short_setup = (bear_power[i] < 0) and \
                          (close[i] < ema_50_1d_aligned[i]) and \
                          low_volatility_regime_aligned[i]
            
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
            # Exit: Bull Power <= 0 OR price below 1d EMA50 OR high volatility regime
            if (bull_power[i] <= 0) or \
               (close[i] <= ema_50_1d_aligned[i]) or \
               (~low_volatility_regime_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: Bear Power >= 0 OR price above 1d EMA50 OR high volatility regime
            if (bear_power[i] >= 0) or \
               (close[i] >= ema_50_1d_aligned[i]) or \
               (~low_volatility_regime_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_BullBearPower_1dTrend_Regime"
timeframe = "6h"
leverage = 1.0