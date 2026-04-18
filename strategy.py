# 4h_Adaptive_RSI_Volume_Strategy_v1
# Strategy: 4h RSI with dynamic thresholds based on volatility regime, combined with volume confirmation
# Rationale: In ranging markets, RSI mean-reverts at extremes; in trending markets, RSI stays extended.
# Uses volatility-adjusted RSI bands to adapt to market regime, reducing false signals.
# Volume confirmation filters low-conviction moves. Designed for 10-25 trades/year.
# Works in bull/bear via adaptive thresholds and volatility filter.

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
    
    # Get daily data for volatility and trend context
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily ATR for volatility regime (14-period)
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.roll(close_1d, 1))
    low_close = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily RSI (14-period) for baseline
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.values
    
    # Volatility-adjusted RSI bands: wider in high vol, narrower in low vol
    atr_ratio = atr_14 / (pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values + 1e-10)
    # Normalize ATR ratio to 0.5-2.0 range for band adjustment
    atr_ratio_adj = np.clip(atr_ratio, 0.5, 2.0)
    # Base RSI levels: 30/70, adjusted by volatility
    rsi_lower = 30 * (2.0 - atr_ratio_adj)  # 30 in low vol, 15 in high vol
    rsi_upper = 70 * atr_ratio_adj          # 70 in low vol, 140 in high vol (capped at 100)
    rsi_upper = np.clip(rsi_upper, 50, 100)  # Cap upper band at 100
    
    # Align daily data to 4h timeframe
    rsi_lower_aligned = align_htf_to_ltf(prices, df_1d, rsi_lower)
    rsi_upper_aligned = align_htf_to_ltf(prices, df_1d, rsi_upper)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # 4h RSI for entry signals
    delta_4h = np.diff(close, prepend=close[0])
    gain_4h = np.where(delta_4h > 0, delta_4h, 0)
    loss_4h = np.where(delta_4h < 0, -delta_4h, 0)
    avg_gain_4h = pd.Series(gain_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss_4h = pd.Series(loss_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs_4h = avg_gain_4h / (avg_loss_4h + 1e-10)
    rsi_4h = 100 - (100 / (1 + rs_4h))
    rsi_4h = rsi_4h.values
    
    # Volume confirmation: 20-period volume average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ausreichend für Indikatoren
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi_4h[i]) or np.isnan(rsi_lower_aligned[i]) or 
            np.isnan(rsi_upper_aligned[i]) or np.isnan(atr_14_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: above average volume
        vol_confirm = volume[i] > 1.3 * vol_ma_20[i]
        
        # RSI-based signals with adaptive bands
        rsi_oversold = rsi_4h[i] < rsi_lower_aligned[i]
        rsi_overbought = rsi_4h[i] > rsi_upper_aligned[i]
        
        if position == 0:
            # Long: RSI oversold + volume confirmation
            if rsi_oversold and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought + volume confirmation
            elif rsi_overbought and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI returns to neutral (50) or overbought
            if rsi_4h[i] >= 50 or rsi_overbought:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI returns to neutral (50) or oversold
            if rsi_4h[i] <= 50 or rsi_oversold:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Adaptive_RSI_Volume_Strategy_v1"
timeframe = "4h"
leverage = 1.0