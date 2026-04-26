#!/usr/bin/env python3
"""
1d_KAMA_Trend_Filter_Volume_Confirmation
Hypothesis: Kaufman Adaptive Moving Average (KAMA) adapts to market noise - 
in trending markets it follows price closely, in ranging markets it flattens.
We use KAMA direction as trend filter combined with volume confirmation 
to capture strong moves while avoiding whipsaws in choppy markets.
Timeframe: 1d, HTF: 1w for regime context.
Target: 7-25 trades/year (30-100 total over 4 years).
Works in both bull (captures trends) and bear (avoids false signals in chop).
"""

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
    
    # Get weekly data for regime filter (optional - can add ADX/chop later if needed)
    # df_1w = get_htf_data(prices, '1w')
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # ER = Efficiency Ratio = |change| / sum(|abs change|)
    # SC = [ER * (fastest SC - slowest SC) + slowest SC]^2
    # KAMA = previous KAMA + SC * (price - previous KAMA)
    
    fast_sc = 2 / (2 + 1)   # for EMA(2)
    slow_sc = 2 / (30 + 1)  # for EMA(30)
    
    # Calculate price changes
    change = np.abs(np.diff(close, prepend=close[0]))
    
    # Efficiency Ratio over 10 periods
    er = np.zeros(n)
    for i in range(10, n):
        # Sum of absolute changes over last 10 periods
        abs_changes = np.abs(np.diff(close[i-9:i+1], prepend=close[i-9]))
        sum_abs = np.sum(abs_changes)
        net_change = abs(close[i] - close[i-10])
        if sum_abs > 0:
            er[i] = net_change / sum_abs
        else:
            er[i] = 0
    
    # Smoothing Constant
    sc = np.zeros(n)
    for i in range(10, n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume confirmation - 20-period volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)  # 1.5x volume MA
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if KAMA not ready
        if np.isnan(kama[i]) or np.isnan(volume_spike[i]):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Trend direction: price above/below KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        if position == 0:
            # Long: Price crosses above KAMA with volume confirmation
            if price_above_kama and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price crosses below KAMA with volume confirmation
            elif price_below_kama and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price crosses back below KAMA
            if price_below_kama:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price crosses back above KAMA
            if price_above_kama:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_Filter_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0