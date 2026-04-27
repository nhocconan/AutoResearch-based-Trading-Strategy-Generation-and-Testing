#!/usr/bin/env python3
"""
1d_Keltner_Breakout_1wTrend_Volume_v1
Hypothesis: Uses 1-day Keltner Channel breakout with 1-week trend filter and volume confirmation.
Keltner Channels (ATR-based) capture volatility breakouts, while weekly trend ensures alignment with higher timeframe momentum.
Volume confirmation filters out weak breakouts. Designed for low trade frequency (10-25 trades/year) to minimize fee drag.
Works in both bull and bear markets by following the weekly trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1-week ATR for Keltner Channels
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range and ATR(20) for 1-week
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Keltner Channels: 20-period EMA ± 2.0 * ATR
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper_keltner = ema20_1w + 2.0 * atr_1w
    lower_keltner = ema20_1w - 2.0 * atr_1w
    
    # 1-week trend: EMA50
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 1.5 * 20-day average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    # Align all 1w indicators to 1d timeframe
    upper_keltner_aligned = align_htf_to_ltf(prices, df_1w, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_1w, lower_keltner)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1w, volume_confirm)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for calculations
    start_idx = 50  # EMA50 needs 50 periods
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_keltner_aligned[i]) or np.isnan(lower_keltner_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_confirm_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper = upper_keltner_aligned[i]
        lower = lower_keltner_aligned[i]
        ema50 = ema50_1w_aligned[i]
        vol_conf = volume_confirm_aligned[i]
        
        if position == 0:
            # Determine trend: price vs EMA50 (1w)
            uptrend = close_val > ema50
            downtrend = close_val < ema50
            
            if uptrend and vol_conf:
                # Long: break above upper Keltner with volume
                if close_val > upper:
                    signals[i] = size
                    position = 1
                    entry_price = close_val
            elif downtrend and vol_conf:
                # Short: break below lower Keltner with volume
                if close_val < lower:
                    signals[i] = -size
                    position = -1
                    entry_price = close_val
        elif position == 1:
            # Exit: price re-enters below upper Keltner or trend reversal
            if close_val < upper:  # Re-enter below upper Keltner
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: price re-enters above lower Keltner or trend reversal
            if close_val > lower:  # Re-enter above lower Keltner
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Keltner_Breakout_1wTrend_Volume_v1"
timeframe = "1d"
leverage = 1.0