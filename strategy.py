#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d KAMA trend direction + 4h RSI mean reversion + volume confirmation.
# Long when 1d KAMA is rising (bullish trend) and 4h RSI < 30 (oversold) with volume > 1.5x 20-period average.
# Short when 1d KAMA is falling (bearish trend) and 4h RSI > 70 (overbought) with volume > 1.5x 20-period average.
# Exit when RSI returns to neutral (40-60 range) or opposite extreme.
# Uses discrete position size 0.25. 1d KAMA provides trend filter from higher timeframe, 4h RSI provides mean-reversion entry.
# Target: 75-200 total trades over 4 years (19-50/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once before loop for KAMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # === Daily Indicators: KAMA ( Kaufman Adaptive Moving Average ) ===
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1d, n=10))
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0)[:len(change)]
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.full_like(close_1d, np.nan, dtype=float)
    kama[29] = close_1d[29]  # seed
    for i in range(30, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # KAMA direction: rising if current > previous, falling if current < previous
    kama_rising = np.roll(kama, 1) < kama
    kama_falling = np.roll(kama, 1) > kama
    kama_rising[0] = False
    kama_falling[0] = False
    
    # Align daily KAMA direction to 4h timeframe
    kama_rising_aligned = align_htf_to_ltf(prices, df_1d, kama_rising.astype(float))
    kama_falling_aligned = align_htf_to_ltf(prices, df_1d, kama_falling.astype(float))
    
    # Get 4h data for RSI and volume
    # RSI (14) on 4h
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Prepend NaN for first value
    rsi = np.concatenate([[np.nan], rsi])
    
    # Volume moving average (20-period) on 4h
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_rising_aligned[i]) or np.isnan(kama_falling_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        kr = kama_rising_aligned[i]
        kf = kama_falling_aligned[i]
        rsi_val = rsi[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if RSI returns to neutral (40-60) or becomes overbought (>70)
            if rsi_val >= 40 or rsi_val > 70:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if RSI returns to neutral (40-60) or becomes oversold (<30)
            if rsi_val <= 60 or rsi_val < 30:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume filter: volume > 1.5x 20-period average
            vol_filter = vol > 1.5 * vol_ma
            
            # LONG: 1d KAMA rising (bullish trend) and 4h RSI < 30 (oversold) with volume confirmation
            if kr and (rsi_val < 30) and vol_filter:
                signals[i] = 0.25
                position = 1
            
            # SHORT: 1d KAMA falling (bearish trend) and 4h RSI > 70 (overbought) with volume confirmation
            elif kf and (rsi_val > 70) and vol_filter:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_1dKAMA_4hRSI_VolumeConfirmation_V1"
timeframe = "4h"
leverage = 1.0