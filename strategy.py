#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with RSI mean reversion and volume confirmation.
# Long when KAMA is rising AND RSI < 30 (oversold) AND 1d volume > 1.2x 20-period average.
# Short when KAMA is falling AND RSI > 70 (overbought) AND 1d volume > 1.2x 20-period average.
# Exit on opposite KAMA direction change.
# Uses discrete position size 0.25. Designed for 1d timeframe to capture medium-term reversals
# in both bull and bear markets by combining trend (KAMA) with mean reversion (RSI) and volume filter.
# Target: 30-100 total trades over 4 years (7-25/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Indicators: KAMA (trend) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # Efficiency ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d, prepend=close_1d[0])), axis=0)  # incorrect, need rolling sum
    # Recalculate properly
    change_1d = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility_1d = pd.Series(close_1d).rolling(window=10, min_periods=10).apply(lambda x: np.sum(np.abs(np.diff(x))), raw=True).values
    er_1d = np.where(volatility_1d != 0, change_1d / volatility_1d, 0)
    # Smoothing constants
    sc_1d = (er_1d * (0.6 - 0.06) + 0.06) ** 2
    # KAMA calculation
    kama_1d = np.zeros_like(close_1d)
    kama_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama_1d[i] = kama_1d[i-1] + sc_1d[i] * (close_1d[i] - kama_1d[i-1])
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # === 1d Indicators: RSI (14) ===
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === 1d Indicators: Volume Spike (volume > 1.2x 20-period average) ===
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.2 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        kama_now = kama_1d_aligned[i]
        kama_prev = kama_1d_aligned[i-1]
        rsi_now = rsi_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if KAMA starts falling (trend change)
            if kama_now < kama_prev:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if KAMA starts rising (trend change)
            if kama_now > kama_prev:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: KAMA rising AND RSI < 30 (oversold) AND volume spike
            if kama_now > kama_prev and rsi_now < 30 and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: KAMA falling AND RSI > 70 (overbought) AND volume spike
            elif kama_now < kama_prev and rsi_now > 70 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "1d_KAMA_RSI_VolumeSpike_V1"
timeframe = "1d"
leverage = 1.0