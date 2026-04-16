#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend direction + RSI(2) mean reversion + volume spike filter.
# KAMA identifies adaptive trend direction (long when price > KAMA, short when price < KAMA).
# RSI(2) provides short-term mean reversion signals (long when RSI(2) < 10, short when RSI(2) > 90).
# Volume spike (>2.0x 20-period 1d average) confirms institutional participation.
# Only trade when KAMA trend and RSI(2) extreme agree, reducing false signals.
# Designed to capture trending moves with pullback entries in both bull and bear markets.
# Uses discrete position size 0.25. Target: 50-120 total trades over 4 years (12-30/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Indicators: KAMA (adaptive trend) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Efficiency Ratio (ER) over 10 periods
    change_1d = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility_1d = np.sum(np.abs(np.diff(close_1d, prepend=close_1d[0])), axis=0) if False else None
    # Proper ER calculation: |close[t] - close[t-10]| / sum(|close[t] - close[t-1]|) over 10 periods
    er_num = np.abs(np.subtract(close_1d[10:], close_1d[:-10]))  # length n-10
    er_den = np.sum(np.abs(np.diff(close_1d, prepend=close_1d[0])[1:]), axis=0) if False else None
    # Vectorized ER calculation
    er = np.full_like(close_1d, np.nan)
    for i in range(10, len(close_1d)):
        num = np.abs(close_1d[i] - close_1d[i-10])
        den = np.sum(np.abs(np.diff(close_1d[i-9:i+1])))
        er[i] = num / den if den != 0 else 0
    
    # Smoothing constants (fast=2, slow=30)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # seed
    for i in range(10, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # === 1d Indicators: RSI(2) ===
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # === 1d Indicators: Volume Spike (>2.0x 20-period average) ===
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (2.0 * vol_ma_1d_aligned)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 30 periods needed for KAMA/RSI)
    warmup = 50
    
    # Track position state and entry price for clarity (though not used for stoploss)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(volume_spike[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if KAMA turns bearish OR RSI(2) > 50 (mean reversion complete)
            if price <= kama_val or rsi_val > 50:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if KAMA turns bullish OR RSI(2) < 50 (mean reversion complete)
            if price >= kama_val or rsi_val < 50:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price > KAMA (bullish trend) AND RSI(2) < 10 (oversold) AND volume spike
            if price > kama_val and rsi_val < 10 and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price < KAMA (bearish trend) AND RSI(2) > 90 (overbought) AND volume spike
            elif price < kama_val and rsi_val > 90 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "1d_KAMA_RSI2_VolumeSpike_V1"
timeframe = "1d"
leverage = 1.0