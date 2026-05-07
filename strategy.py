#!/usr/bin/env python3
name = "6h_Liquidity_Imbalance_Correction_1dLiquidityPools"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for liquidity pools (equal highs/lows)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d liquidity pools: equal highs and lows within 0.1% tolerance
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Find equal highs (resistance liquidity) - within 0.1%
    equal_highs = np.zeros(len(high_1d), dtype=bool)
    equal_lows = np.zeros(len(low_1d), dtype=bool)
    
    tolerance = 0.001  # 0.1%
    
    for i in range(1, len(high_1d)-1):
        # Check if current high is equal to previous or next high (liquidity pool)
        if (abs(high_1d[i] - high_1d[i-1]) / high_1d[i] < tolerance or 
            abs(high_1d[i] - high_1d[i+1]) / high_1d[i] < tolerance):
            equal_highs[i] = True
            
        # Check if current low is equal to previous or next low (liquidity pool)
        if (abs(low_1d[i] - low_1d[i-1]) / low_1d[i] < tolerance or 
            abs(low_1d[i] - low_1d[i+1]) / low_1d[i] < tolerance):
            equal_lows[i] = True
    
    # Liquidity pool levels: price levels where stops are likely clustered
    resistance_liq = np.where(equal_highs, high_1d, np.nan)
    support_liq = np.where(equal_lows, low_1d, np.nan)
    
    # Forward fill liquidity levels (they remain valid until broken)
    resistance_liq_series = pd.Series(resistance_liq)
    support_liq_series = pd.Series(support_liq)
    resistance_liq_ffill = resistance_liq_series.ffill().bfill().values
    support_liq_ffill = support_liq_series.ffill().bfill().values
    
    # 1d trend filter: EMA50
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all to 6h timeframe
    resistance_liq_aligned = align_htf_to_ltf(prices, df_1d, resistance_liq_ffill)
    support_liq_aligned = align_htf_to_ltf(prices, df_1d, support_liq_ffill)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: current volume > 1.5 * 20-period average (moderate for 6h)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 50, 50)
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(resistance_liq_aligned[i]) or np.isnan(support_liq_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above resistance liquidity pool + uptrend + volume
            if close[i] > resistance_liq_aligned[i] and close[i] > ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below support liquidity pool + downtrend + volume
            elif close[i] < support_liq_aligned[i] and close[i] < ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price returns to the opposite liquidity pool (mean reversion to liquidity)
            if position == 1:
                if close[i] < support_liq_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > resistance_liq_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals