# 4h_KAMA_Direction_Trend_Confirmation_v2
# Hypothesis: Use KAMA (Kaufman Adaptive Moving Average) to capture trend direction on 4h,
# filtered by 12h EMA trend and volume confirmation. KAMA adapts to market noise,
# reducing false signals in choppy conditions. Works in both bull and bear by following
# the dominant trend as defined by higher timeframe EMA. Volume ensures breakouts
# have participation. Target: 20-40 trades/year to avoid fee drag.
# Entry: Long when KAMA > 12h EMA and price > KAMA with volume confirmation.
# Short when KAMA < 12h EMA and price < KAMA with volume confirmation.
# Exit: Reverse signal or when price crosses back through KAMA.
# Position size: 0.25 to limit drawdown.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_KAMA_Direction_Trend_Confirmation_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # KAMA parameters
    kama_period = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate Efficiency Ratio (ER) and Smoothing Constant (SC)
    change = np.abs(np.diff(close, kama_period))  # |close(t) - close(t-kama_period)|
    vol = np.sum(np.abs(np.diff(close)), axis=1)  # sum of absolute changes over kama_period
    # Pad the beginning with NaN
    change = np.concatenate([np.full(kama_period, np.nan), change])
    vol = np.concatenate([np.full(kama_period, np.nan), vol])
    # Avoid division by zero
    er = np.where(vol != 0, change / vol, 0)
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[kama_period] = close[kama_period]  # seed
    for i in range(kama_period + 1, n):
        if not np.isnan(kama[i-1]) and not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # 12h EMA for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, kama_period + 1, 34)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama[i]) or np.isnan(ema_12h_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        kama_val = kama[i]
        ema_12h_val = ema_12h_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: KAMA above 12h EMA (uptrend) and price > KAMA with volume
            if kama_val > ema_12h_val and close_val > kama_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: KAMA below 12h EMA (downtrend) and price < KAMA with volume
            elif kama_val < ema_12h_val and close_val < kama_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below KAMA or trend changes
            if close_val < kama_val or kama_val < ema_12h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above KAMA or trend changes
            if close_val > kama_val or kama_val > ema_12h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals