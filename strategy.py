#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA + 1d RSI + Volume Spike - Trend following in trending markets, mean reversion in choppy.
# Uses Kaufman Adaptive Moving Average (KAMA) for trend direction, 1d RSI for overbought/oversold,
# and 1d volume spike for institutional confirmation. Position size 0.25.
# KAMA adapts to market noise - fast in trends, slow in chop.
# Works in bull/bear: adapts to market conditions automatically.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data for RSI and volume spike ===
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 1d volume spike (>2x 20-day average)
    volume_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma20_1d)
    vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)
    volume_spike = vol_1d_current > (2.0 * volume_ma20_1d_aligned)
    
    # === 4h KAMA (10, 2, 30) ===
    # Efficiency Ratio
    change = np.abs(np.diff(close, k=10, prepend=close[:10]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)
    # Vectorized volatility calculation
    volatility_rolling = pd.Series(np.abs(np.diff(close, prepend=close[0]))).rolling(window=10, min_periods=1).sum().values
    er = change / (volatility_rolling + 1e-10)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any required data is not available
        if np.isnan(rsi_1d_aligned[i]) or np.isnan(kama[i]) or np.isnan(volume_ma20_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Get current values
        rsi = rsi_1d_aligned[i]
        vol_spike = volume_spike[i]
        price = close[i]
        kama_val = kama[i]
        
        # Entry conditions
        if position == 0:
            # Long: price > KAMA, RSI < 70 (not overbought), volume spike
            if price > kama_val and rsi < 70 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA, RSI > 30 (not oversold), volume spike
            elif price < kama_val and rsi > 30 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < KAMA or RSI > 70 (overbought)
            if price < kama_val or rsi > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > KAMA or RSI < 30 (oversold)
            if price > kama_val or rsi < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_RSI_VolumeSpike"
timeframe = "4h"
leverage = 1.0