#!/usr/bin/env python3
"""
6h_ElderRay_ZeroLag_MACD_Confluence
Hypothesis: 6h strategy combining Elder Ray (Bull/Bear Power) with zero-lag MACD and weekly trend filter. 
Elder Ray measures bull/bear power relative to EMA13. Zero-lag MACD reduces lag for timely signals. 
Weekly trend filter (price vs weekly EMA20) ensures alignment with higher timeframe momentum. 
Volume confirmation filters low-participation moves. Designed for BTC/ETH robustness in trending and ranging markets. 
Targets 50-150 trades over 4 years (12-37/year) with 0.25 position size. Uses discrete levels to minimize fee drag.
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Get daily data for Elder Ray EMA13
    df_1d = get_htf_data(prices, '1d')
    ema_13_1d = pd.Series(df_1d['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13_1d_aligned
    bear_power = low - ema_13_1d_aligned
    
    # Zero-lag MACD (6h close)
    close_s = pd.Series(close)
    ema_12 = close_s.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close_s.ewm(span=26, adjust=False, min_periods=26).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
    # Zero-lag adjustment: MACD line + (MACD line - Signal line)
    zl_macd = macd_line + (macd_line - signal_line)
    zl_macd_values = zl_macd.values
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Fixed position size to minimize churn
    
    # Warmup: need weekly EMA20 (20), daily EMA13 (13), MACD (max 26,9), vol avg (20)
    start_idx = max(20 + 1, 13 + 1, 26 + 9 + 1, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(ema_13_1d_aligned[i]) or
            np.isnan(zl_macd_values[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        weekly_ema = ema_20_1w_aligned[i]
        zl_macd_val = zl_macd_values[i]
        vol_conf = volume_confirm[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        
        if position == 0:
            # Look for entry: Weekly trend alignment + zero-lag MACD + Elder Ray confirmation + volume
            long_condition = (close_val > weekly_ema and           # Above weekly trend
                            zl_macd_val > 0 and                   # Zero-lag MACD bullish
                            bull_val > 0 and                      # Bull power positive
                            vol_conf)                             # Volume confirmation
            
            short_condition = (close_val < weekly_ema and         # Below weekly trend
                             zl_macd_val < 0 and                  # Zero-lag MACD bearish
                             bear_val < 0 and                     # Bear power negative
                             vol_conf)                            # Volume confirmation
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: weekly trend break or zero-lag MACD bearish crossover
            if (close_val < weekly_ema or 
                (zl_macd_val < 0 and bull_val <= 0)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: weekly trend break or zero-lag MACD bullish crossover
            if (close_val > weekly_ema or 
                (zl_macd_val > 0 and bear_val >= 0)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ElderRay_ZeroLag_MACD_Confluence"
timeframe = "6h"
leverage = 1.0