#!/usr/bin/env python3
"""
1d_KAMA_Regime_VolumeBreakout_v1
Hypothesis: 1d KAMA trend + Bollinger Band squeeze breakout with volume confirmation captures institutional accumulation/distribution phases. Works in bull/bear by using KAMA's adaptive smoothing to avoid whipsaws and BB squeeze to identify low-volatility compression before expansion. Targets 7-25 trades/year with discrete sizing (0.25).
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
    
    # KAMA parameters
    fast_ema = 2
    slow_ema = 30
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # 10-period sum of absolute changes
    # Handle edge case for first 10 values
    change_padded = np.concatenate([np.full(10, np.nan), change])
    volatility_padded = np.concatenate([np.full(10, np.nan), volatility])
    
    # Vectorized ER calculation
    er = np.where(volatility_padded != 0, change_padded / volatility_padded, 0)
    er = np.nan_to_num(er, nan=0.0)
    
    # Smoothing constants
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_stddev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma + bb_std * bb_stddev
    lower_band = sma - bb_std * bb_stddev
    bb_width = (upper_band - lower_band) / sma  # Normalized width
    
    # Bollinger Band squeeze: width < 20th percentile of lookback
    bb_width_lookback = pd.Series(bb_width).rolling(window=50, min_periods=20).mean().values
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=20).quantile(0.2).values
    squeeze = bb_width < bb_width_percentile
    
    # Volume confirmation: 2x average volume (48-period for 2-day average)
    avg_volume = pd.Series(volume).rolling(window=48, min_periods=48).mean().values
    volume_confirmed = volume > 2.0 * avg_volume
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of KAMA initialization, BB period, volume average
    start_idx = max(bb_period, 48) + 10  # +10 for ER lookback
    
    for i in range(start_idx, n):
        kama_val = kama[i]
        close_val = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        sqz = squeeze[i]
        vol_conf = volume_confirmed[i]
        upper = upper_band[i]
        lower = lower_band[i]
        
        # Skip if any data not ready
        if (np.isnan(kama_val) or np.isnan(avg_vol) or np.isnan(sqz) or 
            np.isnan(vol_conf) or np.isnan(upper) or np.isnan(lower)):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Long: price closes above upper band after squeeze with volume
        long_condition = (close_val > upper) and sqz and vol_conf
        # Short: price closes below lower band after squeeze with volume
        short_condition = (close_val < lower) and sqz and vol_conf
        
        # Exit: price returns to middle (SMA)
        long_exit = (position == 1 and close_val <= sma[i])
        short_exit = (position == -1 and close_val >= sma[i])
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "1d_KAMA_Regime_VolumeBreakout_v1"
timeframe = "1d"
leverage = 1.0