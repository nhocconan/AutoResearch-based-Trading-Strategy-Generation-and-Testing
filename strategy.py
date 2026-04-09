#!/usr/bin/env python3
# 12h_daily_camarilla_pivot_volume_v5
# Hypothesis: 12h strategy using daily Camarilla pivot levels (H3/L3) with volume confirmation and ATR filter.
# Long: Price breaks above daily H3, volume > 1.3x 20-period average, and ATR(14) > 0.008*close.
# Short: Price breaks below daily L3, volume > 1.3x 20-period average, and ATR(14) > 0.008*close.
# Exit: Opposite pivot break (L3 for long, H3 for short) or time-based exit (max 3 bars hold).
# Uses tighter Camarilla levels (H3/L3) for more frequent but still selective breakouts,
# volume to avoid low-conviction moves, ATR to ensure sufficient volatility,
# and time-based exit to prevent overtrading and reduce fee drag.
# Target: 12-37 trades/year (50-150 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_daily_camarilla_pivot_volume_v5"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for volatility filter
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift()).abs()
    tr3 = (low_s - close_s.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Get 1d data for Camarilla pivots (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate Camarilla pivots from 1d OHLC
    # Camarilla: H3 = close + 1.125*(high-low), L3 = close - 1.125*(high-low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h3 = close_1d + 1.125 * (high_1d - low_1d)
    camarilla_l3 = close_1d - 1.125 * (high_1d - low_1d)
    
    # Align HTF Camarilla levels to 12h timeframe (wait for completed 1d bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    bars_since_entry = 0
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(atr[i]) or np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or
            np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        # Volatility filter: ATR > 0.8% of price (avoid low-vol chop)
        vol_filter = atr[i] > 0.008 * close[i]
        
        if position == 1:  # Long position
            bars_since_entry += 1
            # Exit: Price breaks below L3 (opposite pivot) OR max 3 bars held
            if low[i] < camarilla_l3_aligned[i] or bars_since_entry >= 3:
                position = 0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            bars_since_entry += 1
            # Exit: Price breaks above H3 (opposite pivot) OR max 3 bars held
            if high[i] > camarilla_h3_aligned[i] or bars_since_entry >= 3:
                position = 0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above H3, volume confirmed, and sufficient volatility
            if (high[i] > camarilla_h3_aligned[i] and volume_confirmed and vol_filter):
                position = 1
                bars_since_entry = 1
                signals[i] = 0.25
            # Short entry: Price breaks below L3, volume confirmed, and sufficient volatility
            elif (low[i] < camarilla_l3_aligned[i] and volume_confirmed and vol_filter):
                position = -1
                bars_since_entry = 1
                signals[i] = -0.25
    
    return signals