#!/usr/bin/env python3
"""
1d_ETF_Rotation_Timing_Signal
Hypothesis: Use 1-week ETF rotation signals (via proxy: BTC dominance + ETH/BTC ratio) 
combined with 1d price action to capture medium-term regime shifts. 
Long when ETH/BTC ratio breaks above 20-period high with rising BTC dominance (alt season). 
Short when ETH/BTC ratio breaks below 20-period low with falling BTC dominance (bitcoin dominance). 
Uses 1-week timeframe for regime filter and 1d for execution. 
Designed for low trade frequency (<25/year) to minimize fee drag in ranging/bear markets.
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
    
    # Get 1d data for ETH/BTC ratio proxy (using price momentum as proxy)
    # Since we don't have ETH/BTC data directly, use 1d close momentum as regime proxy
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Get 1w data for regime filter (BTC dominance proxy via long-term trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 20-period high/low on 1d for breakout signals
    high_20 = pd.Series(close).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(close).rolling(window=20, min_periods=20).min().values
    
    # Calculate ETH/BTC ratio proxy: 1d ROC vs 1w trend
    # Use 1d price position relative to 20-period range as momentum proxy
    price_range = high_20 - low_20
    # Avoid division by zero
    price_range = np.where(price_range == 0, 1e-10, price_range)
    price_position = (close - low_20) / price_range  # 0 to 1
    
    # 1w EMA50 for regime filter (bull/bear)
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    bull_regime = close_1w > ema_50_1w_aligned  # Simplified BTC dominance proxy
    
    # Align 1d indicators to 1d timeframe (no alignment needed for same timeframe)
    price_position_aligned = price_position  # Already on 1d
    
    # Shift 1d data by 1 to use prior day's close for regime (avoid look-ahead)
    price_position_prev = np.roll(price_position_aligned, 1)
    price_position_prev[0] = 0.5  # Neutral start
    
    bull_regime_prev = np.roll(bull_regime, 1)
    bull_regime_prev[0] = True  # Start bullish
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 21
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(price_position_prev[i]) or np.isnan(bull_regime_prev[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: price breaks above 20-period high AND we're in bull regime (BTC dominance rising proxy)
            if (close[i] > high_20[i] and bull_regime_prev[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-period low AND we're in bear regime 
            elif (close[i] < low_20[i] and not bull_regime_prev[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below 20-period low OR regime turns bearish
            if (close[i] < low_20[i] or not bull_regime_prev[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above 20-period high OR regime turns bullish
            if (close[i] > high_20[i] or bull_regime_prev[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_ETF_Rotation_Timing_Signal"
timeframe = "1d"
leverage = 1.0