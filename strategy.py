# 4h_KAMA_Trend_With_Volume_Confirmation
# Hypothesis: 4h Kaufman Adaptive Moving Average (KAMA) provides trend direction that adapts to market noise
# KAMA reduces whipsaw in choppy markets while capturing trends. Combined with volume confirmation
# and ATR-based stop loss, this aims to reduce false signals and improve risk-adjusted returns.
# Designed for 4h timeframe targeting 75-200 total trades over 4 years (19-50/year).
# Works in bull/bear via adaptive trend filtering and volume confirmation.

name = "4h_KAMA_Trend_With_Volume_Confirmation"
timeframe = "4h"
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
    
    # Calculate KAMA ( Kaufman Adaptive Moving Average )
    def calculate_kama(close, period=10, fast=2, slow=30):
        # Direction
        change = np.abs(close - np.roll(close, period))
        change[0:period] = np.nan
        
        # Volatility
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if hasattr(np, 'sum') else np.nansum(np.abs(np.diff(close)))
        # Correct volatility calculation
        volatility = np.zeros_like(close)
        for i in range(1, len(close)):
            volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
        volatility = pd.Series(volatility).rolling(window=period, min_periods=period).sum().values
        
        # Efficiency Ratio
        er = np.zeros_like(close)
        mask = volatility > 0
        er[mask] = change[mask] / volatility[mask]
        
        # Smoothing Constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        
        # KAMA calculation
        kama = np.full_like(close, np.nan)
        kama[0] = close[0]
        for i in range(1, len(close)):
            if not np.isnan(kama[i-1]) and not np.isnan(sc[i]):
                kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
            else:
                kama[i] = kama[i-1] if not np.isnan(kama[i-1]) else close[i]
        return kama
    
    # 4h data for KAMA and other indicators
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate KAMA on 4h data
    kama_4h = calculate_kama(df_4h['close'].values, 10, 2, 30)
    kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
    
    # Volume confirmation: volume > 1.3 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.3)
    
    # ATR for stop loss and position sizing adjustment
    def calculate_atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
        return atr
    
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_4h_aligned[i]) or np.isnan(volume_ma[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above KAMA with volume confirmation
            if (close[i] > kama_4h_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA with volume confirmation
            elif (close[i] < kama_4h_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price crosses below KAMA or ATR-based stop
            if (close[i] < kama_4h_aligned[i]) or (close[i] < close[i-1] - 1.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price crosses above KAMA or ATR-based stop
            if (close[i] > kama_4h_aligned[i]) or (close[i] > close[i-1] + 1.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals