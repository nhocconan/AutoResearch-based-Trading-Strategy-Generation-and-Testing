#!/usr/bin/env python3
"""
12h_TRIX_ZeroCross_1dTrend_VolumeSpike
Hypothesis: Uses TRIX (15-period) zero-cross for momentum signals on 12h timeframe, filtered by daily EMA34 trend and volume spike (>2.0x 20-period avg). Enters long when TRIX crosses above zero AND daily trend is up AND volume spike; enters short when TRIX crosses below zero AND daily trend is down AND volume spike. Exits when TRIX reverses back across zero. TRIX filters noise better than MACD in choppy markets, and the 1d trend filter ensures alignment with higher timeframe structure. Volume confirmation avoids weak breakouts. Designed for low trade frequency (target: 50-150 total trades over 4 years) with 0.25 position size to manage drawdown in bear markets.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA34 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate TRIX on primary timeframe (12h close)
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15) - then percent change
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = 100 * (ema3.pct_change())  # Percent change of triple EMA
    trix_values = trix.values
    
    # Calculate prior TRIX for zero-cross detection
    prior_trix = np.roll(trix_values, 1)
    prior_trix[0] = np.nan  # First value invalid
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need TRIX (15*3=45 for stability), 1d EMA34 (34), volume avg (20)
    start_idx = max(45, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(trix_values[i]) or np.isnan(prior_trix[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        trix_val = trix_values[i]
        prior_trix_val = prior_trix[i]
        ema_val = ema_34_1d_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: TRIX zero-cross with 1d trend filter AND volume
            # Long: TRIX crosses above zero AND 1d uptrend AND volume
            long_condition = (prior_trix_val <= 0) and (trix_val > 0) and (close[i] > ema_val) and vol_conf
            # Short: TRIX crosses below zero AND 1d downtrend AND volume
            short_condition = (prior_trix_val >= 0) and (trix_val < 0) and (close[i] < ema_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when TRIX crosses back below zero
            exit_condition = (prior_trix_val >= 0) and (trix_val <= 0)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when TRIX crosses back above zero
            exit_condition = (prior_trix_val <= 0) and (trix_val >= 0)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_TRIX_ZeroCross_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0