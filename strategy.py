#!/usr/bin/env python3
"""
1d_KAMA_Regime_VolumeBreakout_v2
Hypothesis: Kaufman Adaptive Moving Average (KAMA) trend direction + volume spike + Bollinger Band squeeze regime filter captures strong momentum moves while avoiding whipsaws in ranging markets. Works in bull/bear via KAMA's adaptive nature and regime filter. Designed for 1d to target 7-25 trades/year with discrete sizing (0.25).
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
    
    # Load 1d data ONCE before loop for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # KAMA parameters
    fast = 2
    slow = 30
    
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # 10-period volatility
    # Fix: volatility needs rolling sum
    volatility_series = pd.Series(np.abs(np.diff(close))).rolling(window=10, min_periods=1).sum().values
    volatility_series = np.concatenate([[np.nan], volatility_series])  # align with change
    er = np.where(volatility_series > 0, change / volatility_series, 0)
    
    # Smoothing Constant (SC)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Align KAMA to 1d (wait for completed 1d bar)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Bollinger Bands (20, 2) for regime filter
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma + (bb_std * std_dev)
    lower_band = sma - (bb_std * std_dev)
    bb_width = (upper_band - lower_band) / sma
    
    # Bollinger Band Width percentile (50-period) for squeeze detection
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    
    # Average volume for confirmation (24-period SMA = 24 * 1h = 1d equivalent)
    avg_volume = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of KAMA(10), BB(20,50), volume(24)
    start_idx = max(30, 50, 24)  # KAMA needs ~30 for stability
    
    for i in range(start_idx, n):
        close_val = close[i]
        vol = volume[i]
        kama_val = kama_aligned[i]
        bb_width_pct = bb_width_percentile[i]
        avg_vol = avg_volume[i]
        
        # Skip if any data not ready
        if (np.isnan(kama_val) or np.isnan(bb_width_pct) or np.isnan(avg_vol)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = vol > 1.5 * avg_vol
        
        # Regime filter: Bollinger Band Width < 30th percentile = squeeze (low volatility)
        # We want to trade breakouts FROM squeeze, so we require recent squeeze
        squeeze_condition = bb_width_pct < 0.3
        
        # Trend filter: price vs KAMA
        uptrend = close_val > kama_val
        downtrend = close_val < kama_val
        
        # Long: price above KAMA with volume confirmation and recent squeeze
        long_condition = uptrend and volume_confirmed and squeeze_condition
        # Short: price below KAMA with volume confirmation and recent squeeze
        short_condition = downtrend and volume_confirmed and squeeze_condition
        
        # Exit: price crosses KAMA in opposite direction
        long_exit = position == 1 and close_val < kama_val
        short_exit = position == -1 and close_val > kama_val
        
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

name = "1d_KAMA_Regime_VolumeBreakout_v2"
timeframe = "1d"
leverage = 1.0