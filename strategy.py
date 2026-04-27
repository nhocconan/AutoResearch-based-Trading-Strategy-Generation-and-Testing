#!/usr/bin/env python3
"""
1h_Keltner_Channel_Squeeze_Momentum_v1
Hypothesis: Keltner Channel squeeze (low volatility) followed by expansion with momentum captures breakouts in both bull and bear markets. 
Uses 4h trend filter to align with higher timeframe direction and volume confirmation to filter false signals. 
Designed for low trade frequency (target: 60-150 trades over 4 years) to minimize fee drag. 
Works in bull markets by catching momentum continuations and in bear markets by catching sharp reversals after low volatility periods.
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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate Keltner Channel (20, 2.0) on 1h
    # Middle line: 20-period EMA
    # Upper/lower: EMA ± 2.0 * ATR(10)
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    tr = np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    kc_upper = ema20 + 2.0 * atr10
    kc_lower = ema20 - 2.0 * atr10
    
    # Bollinger Bands (20, 2.0) for squeeze detection
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma20 + 2.0 * std20
    bb_lower = sma20 - 2.0 * std20
    
    # Squeeze condition: BB inside KC (low volatility)
    squeeze = (bb_upper <= kc_upper) & (bb_lower >= kc_lower)
    
    # Momentum: price change over 3 periods
    mom = close - np.roll(close, 3)
    mom[0:3] = 0
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    # 4h trend filter: EMA50
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Align indicators to 1h
    squeeze_aligned = align_htf_to_ltf(prices, pd.DataFrame({'squeeze': squeeze}), squeeze)[:, 0]
    mom_aligned = align_htf_to_ltf(prices, pd.DataFrame({'mom': mom}), mom)[:, 0]
    volume_confirm_aligned = align_htf_to_ltf(prices, pd.DataFrame({'volume_confirm': volume_confirm}), volume_confirm)[:, 0]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # Position size: 20% of capital
    
    # Warmup: need EMA20 (20), ATR10 (10), BBands (20), mom (3), vol_avg (20)
    start_idx = max(20, 10, 20, 3, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(squeeze_aligned[i]) or np.isnan(mom_aligned[i]) or 
            np.isnan(volume_confirm_aligned[i]) or np.isnan(ema50_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        is_squeeze = squeeze_aligned[i]
        momentum = mom_aligned[i]
        vol_conf = volume_confirm_aligned[i]
        ema50_4h = ema50_4h_aligned[i]
        
        if position == 0:
            # Look for squeeze breakout with momentum and volume
            if not is_squeeze and vol_conf:  # Squeeze just ended
                # Long: price above KC upper with upward momentum and uptrend
                if close_val > kc_upper[i] and momentum > 0 and close_val > ema50_4h:
                    signals[i] = size
                    position = 1
                # Short: price below KC lower with downward momentum and downtrend
                elif close_val < kc_lower[i] and momentum < 0 and close_val < ema50_4h:
                    signals[i] = -size
                    position = -1
        elif position == 1:
            # Exit: price crosses below KC middle or momentum reverses
            if close_val < ema20[i] or momentum < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: price crosses above KC middle or momentum reverses
            if close_val > ema20[i] or momentum > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Keltner_Channel_Squeeze_Momentum_v1"
timeframe = "1h"
leverage = 1.0