#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_RSI_Momentum
Hypothesis: KAMA adapts to market noise, providing a smooth trend line that reduces whipsaws. Combined with RSI momentum (40-60 for mean reversion avoidance) and 1-week ADX trend filter (>25 for trending markets), this strategy captures sustained moves while avoiding choppy markets. Works in bull markets via KAMA-up/RSI>50 and in bear markets via KAMA-down/RSI<50. Targets 10-20 trades/year on 1d to minimize fee drag.
"""

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
    
    # Get 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate KAMA on close
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(close - np.roll(close, 10))
    change[0:10] = np.nan  # not enough data
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # will fix below
    
    # Proper volatility calculation: sum of absolute changes over 10 periods
    volatility = np.zeros_like(close)
    for i in range(n):
        if i < 10:
            volatility[i] = np.nan
        else:
            volatility[i] = np.sum(np.abs(np.diff(close[i-9:i+1])))
    
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (0.6667 - 0.0645) + 0.0645) ** 2
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # RSI (14-period)
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)  # same length as close
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # ADX (14-period) on 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w - np.roll(low_1w, 1))
    tr2 = np.abs(low_1w - np.roll(high_1w, 1))
    tr3 = np.abs(high_1w - np.roll(high_1w, 1))
    tr_1w = np.maximum(np.maximum(tr1, tr2), tr3)
    tr_1w[0] = high_1w[0] - low_1w[0]
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w),
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)),
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr14 = pd.Series(tr_1w).rolling(window=14, min_periods=14).sum().values
    dm_plus14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI-
    di_plus = np.where(tr14 != 0, 100 * dm_plus14 / tr14, 0)
    di_minus = np.where(tr14 != 0, 100 * dm_minus14 / tr14, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1w indicators to 1d
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume confirmation: volume > 1.5 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for KAMA, RSI, ADX, and volume MA
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        kama_val = kama[i]
        rsi_val = rsi[i]
        adx_val = adx_1w_aligned[i]
        vol_filter_val = vol_filter[i]
        
        if position == 0:
            # Long: KAMA up (close > KAMA) + RSI > 50 (bullish momentum) + ADX > 25 (trending) + volume
            if close[i] > kama_val and rsi_val > 50 and adx_val > 25 and vol_filter_val:
                signals[i] = size
                position = 1
            # Short: KAMA down (close < KAMA) + RSI < 50 (bearish momentum) + ADX > 25 (trending) + volume
            elif close[i] < kama_val and rsi_val < 50 and adx_val > 25 and vol_filter_val:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: KAMA down OR RSI < 40 (loss of momentum) OR ADX < 20 (losing trend)
            if close[i] < kama_val or rsi_val < 40 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: KAMA up OR RSI > 60 (loss of momentum) OR ADX < 20 (losing trend)
            if close[i] > kama_val or rsi_val > 60 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_Trend_With_RSI_Momentum"
timeframe = "1d"
leverage = 1.0