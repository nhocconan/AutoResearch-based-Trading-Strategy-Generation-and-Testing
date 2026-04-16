#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend direction + 1d RSI(2) mean reversion + 1w volume spike filter.
# Long when 1d KAMA rising AND RSI(2) < 10 AND 1w volume > 2.0x 20-period average.
# Short when 1d KAMA falling AND RSI(2) > 90 AND 1w volume > 2.0x 20-period average.
# Exit when RSI(2) crosses 50 (mean reversion complete).
# Uses discrete position size 0.25. KAMA filters false signals in chop, RSI(2) catches extreme reversals,
# 1w volume spike ensures institutional participation. Target: 50-100 trades over 4 years (12-25/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Get 1d data once before loop for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: KAMA(10,2,30) for trend direction ===
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_dir = np.diff(kama, prepend=0) > 0  # True if rising
    
    # === 1d Indicators: RSI(2) for mean reversion ===
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(np.isnan(rsi), 50, rsi)
    
    # Align 1d indicators to lower timeframe
    kama_dir_aligned = align_htf_to_ltf(prices, df_1d, kama_dir.astype(float))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Get 1w data once before loop for volume filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    volume_1w = df_1w['volume'].values
    vol_ma_20 = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_dir_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        kama_up = kama_dir_aligned[i] > 0.5  # KAMA rising
        rsi_val = rsi_aligned[i]
        vol_ma_val = vol_ma_aligned[i]
        price = close[i]
        # Use current bar's volume approximation from 1d volume (since 1w volume not available per bar)
        # We'll use price change as proxy for volume urgency when 1w data is aligned
        vol_filter = vol_ma_val > 0 and (close[i] > close[i-1] if i>0 else True)  # simple uptrend proxy
        
        # Volume filter: 1w volume > 2.0x 20-period average (using aligned data)
        vol_spike = vol_ma_val > 0 and (volume_1w[min(i//(7*24*4), len(volume_1w)-1)] > 2.0 * vol_ma_val) if len(volume_1w) > i//(7*24*4) else False
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if RSI(2) crosses above 50 (mean reversion complete)
            if rsi_val >= 50:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if RSI(2) crosses below 50 (mean reversion complete)
            if rsi_val <= 50:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: KAMA rising AND RSI(2) < 10 AND 1w volume spike
            if kama_up and rsi_val < 10 and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: KAMA falling AND RSI(2) > 90 AND 1w volume spike
            elif not kama_up and rsi_val > 90 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "1d_KAMA10_2_30_RSI2_VolumeSpike_1w_V1"
timeframe = "1d"
leverage = 1.0