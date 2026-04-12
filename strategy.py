# 4h_1d_kama_volatility_regime
# Hypothesis: 4-hour KAMA trend with 1-day volatility regime filter and volume confirmation
# Uses KAMA (Kaufman Adaptive Moving Average) for adaptive trend following, combined with
# 1-day ATR-based volatility regime (high volatility = trend following, low volatility = avoid)
# Volume confirmation filters low-quality breakouts. Designed to work in both bull and bear
# markets by avoiding choppy low-volatility periods and catching strong trends.
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.

name = "4h_1d_kama_volatility_regime"
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
    
    # Get daily data for KAMA, volatility regime, and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # KAMA (Kaufman Adaptive Moving Average) - 10-period ER, 2 and 30 SC
    def kama(close, er_period=10, fast_sc=2, slow_sc=30):
        change = np.abs(np.subtract(close, np.roll(close, er_period)))
        volatility = np.sum(np.abs(np.subtract(np.roll(close, 1), close)), axis=0) if len(close.shape) > 0 else np.sum(np.abs(np.subtract(np.roll(close, 1), close)))
        # For 1D array, calculate rolling volatility
        volatility_arr = np.zeros_like(close)
        for i in range(er_period, len(close)):
            volatility_arr[i] = np.sum(np.abs(np.subtract(close[i-er_period+1:i+1], np.roll(close[i-er_period+1:i+1], 1))))
        volatility_arr[:er_period] = np.nan
        er = np.divide(change, volatility_arr, out=np.zeros_like(change), where=volatility_arr!=0)
        sc = np.power(er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1), 2)
        kama_out = np.zeros_like(close)
        kama_out[0] = close[0]
        for i in range(1, len(close)):
            kama_out[i] = kama_out[i-1] + sc[i] * (close[i] - kama_out[i-1])
        return kama_out
    
    # Calculate KAMA on daily close
    kama_1d = kama(close_1d, er_period=10, fast_sc=2, slow_sc=30)
    
    # Volatility regime: 1-day ATR ratio (current ATR / 20-period average ATR)
    tr1 = np.abs(np.subtract(high_1d, low_1d))
    tr2 = np.abs(np.subtract(high_1d, np.roll(close_1d, 1)))
    tr3 = np.abs(np.subtract(low_1d, np.roll(close_1d, 1)))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_current = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    atr_ma = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    atr_ratio = np.divide(atr_current, atr_ma, out=np.ones_like(atr_current), where=atr_ma!=0)
    
    # ATR for volatility filter (avoid extremely low volatility)
    atr_filter = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align KAMA, ATR ratio, and ATR filter to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    atr_filter_aligned = align_htf_to_ltf(prices, df_1d, atr_filter)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(atr_filter_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in high volatility regimes (ATR ratio > 0.8) to avoid chop
        vol_regime_ok = atr_ratio_aligned[i] > 0.8
        
        # Long entry: price above KAMA with volume and volatility filter
        if (close[i] > kama_aligned[i] and vol_confirm[i] and 
            atr_filter_aligned[i] > 0 and vol_regime_ok and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price below KAMA with volume and volatility filter
        elif (close[i] < kama_aligned[i] and vol_confirm[i] and 
              atr_filter_aligned[i] > 0 and vol_regime_ok and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: reverse signal
        elif position == 1 and close[i] < kama_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > kama_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals