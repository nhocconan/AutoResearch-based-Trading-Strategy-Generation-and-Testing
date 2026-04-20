#!/usr/bin/env python3
"""
1h_4d_RSI_Trend_With_Volume_Regime_v1
Concept: 1h momentum with 4h trend filter and volume regime confirmation.
- Long when 1h RSI crosses above 50, 4h EMA50 is rising, and volume > 1.5x average
- Short when 1h RSI crosses below 50, 4h EMA50 is falling, and volume > 1.5x average
- Exit when RSI crosses back to 50 (mean reversion)
- Uses 4h for trend direction, 1h only for entry timing
- Conservative sizing (0.20) to manage drawdown
- Works in bull/bear: RSI mean reversion works in ranges, trend filter captures trends
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4d_RSI_Trend_With_Volume_Regime_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # === 4h: EMA50 trend filter ===
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_rising = ema50_4h > np.roll(ema50_4h, 1)  # rising if current > previous
    ema50_4h_falling = ema50_4h < np.roll(ema50_4h, 1)  # falling if current < previous
    
    # Align 4h EMA trend to 1h timeframe
    ema50_4h_rising_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h_rising)
    ema50_4h_falling_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h_falling)
    
    # === 1h: RSI calculation ===
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss > 0, avg_loss, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    # === 1h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for EMA50 and RSI
    
    for i in range(start_idx, n):
        # Get values
        rsi_val = rsi[i]
        ema50_rising = ema50_4h_rising_aligned[i]
        ema50_falling = ema50_4h_falling_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(rsi_val) or np.isnan(ema50_rising) or np.isnan(ema50_falling) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI crosses above 50, 4h EMA50 rising, volume confirmation
            rsi_cross_up = rsi_val > 50 and rsi[i-1] <= 50
            vol_confirm = vol_ratio_val > 1.5  # Volume above average
            
            if rsi_cross_up and ema50_rising and vol_confirm:
                signals[i] = 0.20
                position = 1
            # Short: RSI crosses below 50, 4h EMA50 falling, volume confirmation
            elif rsi_val < 50 and rsi[i-1] >= 50 and ema50_falling and vol_confirm:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: RSI crosses back to 50 (mean reversion)
            if rsi_val < 50 and rsi[i-1] >= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: RSI crosses back to 50 (mean reversion)
            if rsi_val > 50 and rsi[i-1] <= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals