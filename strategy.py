#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrend_RegimeFilter
Hypothesis: Elder Ray (Bull/Bear Power) with 1d EMA trend filter and Bollinger Bandwidth regime.
Long when Bear Power crosses above zero (bulls gaining control) in 1d uptrend + low volatility (chop regime).
Short when Bull Power crosses below zero (bears gaining control) in 1d downtrend + low volatility.
Uses discrete sizing (0.25) and ATR trailing stop (2.0) to limit trades (~12-30/year) and minimize fee drag.
Designed for BTC/ETH to work in bull/bear via trend-following with volatility regime filter.
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
    
    # 1d data for HTF trend filter and Elder Ray
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (using 1d EMA13)
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema_13_1d  # Bull Power
    bear_power = low_1d - ema_13_1d   # Bear Power
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Bollinger Bandwidth (20,2) on 1d for regime filter (chop = low volatility)
    bb_middle = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2.0 * bb_std
    bb_lower = bb_middle - 2.0 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_middle
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width)
    
    # ATR for stop loss (14-period) on primary timeframe
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need EMA34 (34), EMA13 (13), BB width (20), ATR (14)
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(bb_width_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Regime filter: low volatility (chop regime) = BB width below 50th percentile (adaptive threshold)
        # Use rolling 50-period percentile of BB width to define choppy vs trending
        if i >= 50:
            bb_width_percentile = pd.Series(bb_width_aligned[:i+1]).rolling(window=50, min_periods=10).quantile(0.5).iloc[-1]
            chop_regime = bb_width_aligned[i] < bb_width_percentile
        else:
            chop_regime = True  # default to chop regime early on
        
        if position == 0:
            # Long: Bear Power crosses above zero (bulls gaining control) in 1d uptrend + chop regime
            long_signal = (bear_power_aligned[i-1] <= 0 and bear_power_aligned[i] > 0) and \
                         (close_1d_aligned := align_htf_to_ltf(prices, df_1d, close_1d)[i]) > ema_34_1d_aligned[i] and \
                         chop_regime
            # Short: Bull Power crosses below zero (bears gaining control) in 1d downtrend + chop regime
            short_signal = (bull_power_aligned[i-1] >= 0 and bull_power_aligned[i] < 0) and \
                          (close_1d_aligned := align_htf_to_ltf(prices, df_1d, close_1d)[i]) < ema_34_1d_aligned[i] and \
                          chop_regime
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: Bear Power crosses below zero OR trend turns down OR ATR stoploss hit
            if (bear_power_aligned[i-1] > 0 and bear_power_aligned[i] <= 0) or \
               (curr_close < ema_34_1d_aligned[i]) or \
               (curr_close < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: Bull Power crosses above zero OR trend turns up OR ATR stoploss hit
            if (bull_power_aligned[i-1] < 0 and bull_power_aligned[i] >= 0) or \
               (curr_close > ema_34_1d_aligned[i]) or \
               (curr_close > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_BullBearPower_1dTrend_RegimeFilter"
timeframe = "6h"
leverage = 1.0