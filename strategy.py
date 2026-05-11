#!/usr/bin/env python3
"""
4h_Vortex_Trend_12hVol
Hypothesis: The Vortex Indicator identifies strong trend direction (VI+ > VI- for uptrend, VI- > VI+ for downtrend). Combined with 12h volume confirmation (volume > 1.5x median) and ATR-based stoploss, this strategy captures sustained trends in both bull and bear markets. The Vortex Indicator is less prone to whipsaws than simple moving average crossovers. Target: 20-50 trades/year to avoid fee drag.
"""

name = "4h_Vortex_Trend_12hVol"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data for volume filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    volume_4h = prices['volume'].values
    
    # --- Vortex Indicator (VI) on 4h ---
    # True Range
    tr1 = np.abs(high_4h - low_4h)
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    
    # VM+ and VM-
    vm_plus = np.abs(high_4h - np.roll(low_4h, 1))
    vm_minus = np.abs(low_4h - np.roll(high_4h, 1))
    vm_plus[0] = np.abs(high_4h[0] - low_4h[0])  # first bar
    vm_minus[0] = np.abs(low_4h[0] - high_4h[0])  # first bar
    
    # Sum over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    vm_plus_sum = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    vm_minus_sum = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values
    
    vi_plus = vm_plus_sum / tr_sum
    vi_minus = vm_minus_sum / tr_sum
    
    # --- 12h Volume Filter: volume > 1.5x median of last 28 periods ---
    vol_12h = df_12h['volume'].values
    vol_median_12h = pd.Series(vol_12h).rolling(window=28, min_periods=14).median().values
    vol_threshold_12h = vol_median_12h * 1.5
    vol_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_12h)
    vol_threshold_aligned = align_htf_to_ltf(prices, df_12h, vol_threshold_12h)
    
    # --- ATR for stoploss ---
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 30  # for Vortex and ATR
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or 
            np.isnan(vol_12h_aligned[i]) or np.isnan(vol_threshold_aligned[i]) or 
            np.isnan(atr_4h[i])):
            if position != 0:
                # Check stoploss
                if position == 1 and close_4h[i] <= entry_price - 2.5 * atr_4h[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_4h[i] >= entry_price + 2.5 * atr_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.30 if position == 1 else -0.30
            continue
        
        # Determine trend direction from Vortex
        bullish_trend = vi_plus[i] > vi_minus[i]
        bearish_trend = vi_minus[i] > vi_plus[i]
        
        # Volume filter: 12h volume above threshold
        vol_ok = vol_12h_aligned[i] > vol_threshold_aligned[i]
        
        if position == 0:
            # Look for entries only with volume confirmation
            if bullish_trend and vol_ok:
                # Long: VI+ > VI- + volume confirmation
                signals[i] = 0.30
                position = 1
                entry_price = close_4h[i]
            elif bearish_trend and vol_ok:
                # Short: VI- > VI+ + volume confirmation
                signals[i] = -0.30
                position = -1
                entry_price = close_4h[i]
        else:
            # Update stoploss and check exits
            if position == 1:
                # Stoploss
                if close_4h[i] <= entry_price - 2.5 * atr_4h[i]:
                    signals[i] = 0.0
                    position = 0
                # Exit: trend reversal (VI- > VI+)
                elif vi_minus[i] > vi_plus[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.30
            elif position == -1:
                # Stoploss
                if close_4h[i] >= entry_price + 2.5 * atr_4h[i]:
                    signals[i] = 0.0
                    position = 0
                # Exit: trend reversal (VI+ > VI-)
                elif vi_plus[i] > vi_minus[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.30
    
    return signals