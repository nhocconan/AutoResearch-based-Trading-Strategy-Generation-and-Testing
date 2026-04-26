#!/usr/bin/env python3
"""
4h_KAMA_Regime_Reversal_v1
Hypothesis: 4h timeframe captures medium-term swings. KAMA adapts to market noise, identifying true trend changes. Combined with Bollinger Band squeeze regime filter and volume confirmation, it avoids whipsaws in ranging markets while catching reversals at extremes. Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend) by trading mean reversion only during low volatility (chop) regimes. Targets 20-40 trades/year for optimal fee efficiency on BTC/ETH.
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
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (UTC 8-20) for institutional activity
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate KAMA on 4h close
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)
    er = np.zeros_like(change)
    mask = volatility != 0
    er[mask] = change[mask] / volatility[mask]
    er = np.concatenate([np.full(10, np.nan), er])
    
    # Smoothing constants
    fastest = 2.0 / (2 + 1)   # EMA(2)
    slowest = 2.0 / (30 + 1)  # EMA(30)
    sc = (er * (fastest - slowest) + slowest) ** 2
    sc = np.concatenate([np.full(10, np.nan), sc[:-10]])  # align with close
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate Bollinger Bands (20, 2) for squeeze regime
    close_s = pd.Series(close)
    basis = close_s.rolling(window=20, min_periods=20).mean().values
    dev = close_s.rolling(window=20, min_periods=20).std().values
    upper_band = basis + 2.0 * dev
    lower_band = basis - 2.0 * dev
    bb_width = (upper_band - lower_band) / basis
    
    # Bollinger Band squeeze: low volatility regime (width < 20-period percentile 20)
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    bb_width_percentile = pd.Series(bb_width).rolling(window=20, min_periods=20).quantile(0.2).values
    low_volatility = bb_width < bb_width_percentile  # squeeze regime
    
    # Volume average (20-period) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 40  # ensures KAMA, BB, volume MA ready
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(kama[i]) or 
            np.isnan(basis[i]) or 
            np.isnan(vol_ma[i]) or
            not in_session[i]):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        kama_val = kama[i]
        basis_val = basis[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        low_vol_val = low_volatility[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = vol_val > 1.3 * vol_ma_val
        
        if position == 0:
            # Long: price below KAMA (dip) in low volatility regime with volume confirmation
            long_signal = (close_val < kama_val) and low_vol_val and volume_confirmed
            # Short: price above KAMA (rally) in low volatility regime with volume confirmation
            short_signal = (close_val > kama_val) and low_vol_val and volume_confirmed
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price crosses above KAMA (trend resumption) OR volatility expands
            if close_val > kama_val or not low_vol_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price crosses below KAMA (trend resumption) OR volatility expands
            if close_val < kama_val or not low_vol_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_KAMA_Regime_Reversal_v1"
timeframe = "4h"
leverage = 1.0