#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_kama_rsi_momentum_v1"
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Calculate 1d KAMA (Kaufman Adaptive Moving Average)
    close_1d = df_1d['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i == 0:
            er[i] = 0
        else:
            er[i] = change[i] / (np.sum(volatility[max(0, i-9):i+1]) + 1e-10)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate 1d RSI
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Shift by 1 to use only completed 1d bars
    kama = np.roll(kama, 1)
    rsi = np.roll(rsi, 1)
    kama[0] = np.nan
    rsi[0] = np.nan
    
    # Align 1d indicators to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate 4h RSI for entry timing
    delta_4h = np.diff(close, prepend=close[0])
    gain_4h = np.where(delta_4h > 0, delta_4h, 0)
    loss_4h = np.where(delta_4h < 0, -delta_4h, 0)
    avg_gain_4h = pd.Series(gain_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_4h = pd.Series(loss_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_4h = avg_gain_4h / (avg_loss_4h + 1e-10)
    rsi_4h = 100 - (100 / (1 + rs_4h))
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or
            np.isnan(rsi_4h[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.3 * vol_ma
        
        # Determine trend from 1d KAMA slope
        kama_slope = kama_aligned[i] - kama_aligned[i-1] if i > 0 else 0
        
        # Long conditions: Uptrend (KAMA rising) + RSI not oversold + volume
        long_signal = volume_confirmed and (kama_slope > 0) and (rsi_4h[i] > 30) and (rsi_4h[i] < 70)
        
        # Short conditions: Downtrend (KAMA falling) + RSI not overbought + volume
        short_signal = volume_confirmed and (kama_slope < 0) and (rsi_4h[i] > 30) and (rsi_4h[i] < 70)
        
        # Exit when trend changes
        exit_long = position == 1 and kama_slope <= 0
        exit_short = position == -1 and kama_slope >= 0
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 4h momentum strategy using 1d KAMA for trend direction and 4h RSI for entry timing.
# Uses Kaufman Adaptive Moving Average (KAMA) on daily timeframe to identify adaptive trend.
# Enters long when daily KAMA is rising (uptrend) and 4h RSI is between 30-70 (not extreme)
# with volume confirmation (>1.3x average). Enters short when daily KAMA is falling
# (downtrend) with same RSI and volume conditions. Exits when KAMA slope changes direction.
# KAMA adapts to market noise, reducing false signals in choppy conditions. Volume
# confirmation ensures participation. Designed for 4h timeframe to target 20-50 trades/year
# (80-200 total over 4 years) to minimize fee drag. Works in both bull and bear markets
# by following the adaptive trend on higher timeframe.