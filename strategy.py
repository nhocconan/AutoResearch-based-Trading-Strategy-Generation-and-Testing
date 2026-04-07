#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h KAMA Trend + 1d Volume Profile + RSI Filter
# Hypothesis: Kaufman Adaptive Moving Average (KAMA) adapts to market noise,
# reducing whipsaws in choppy markets. Combined with daily volume profile
# (high volume nodes as support/resistance) and RSI extremes, it captures
# trend continuation in both bull and bear markets.
# Target: 15-25 trades/year (60-100 total over 4 years).

name = "6h_kama_volprofile_rsi_filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume profile and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d RSI(14)
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_values = rsi_1d.values
    
    # Calculate volume profile: identify high volume nodes (top 20% volume days)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=10).mean().values
    vol_spike_1d = volume_1d > (1.5 * vol_ma_20)
    
    # Identify support/resistance from high volume days
    # For simplicity, use recent high volume day's high/low as S/R
    vol_high_idx = np.where(vol_spike_1d)[0]
    if len(vol_high_idx) > 0:
        last_vol_high_idx = vol_high_idx[-1]
        vol_support = low_1d[last_vol_high_idx]
        vol_resistance = high_1d[last_vol_high_idx]
    else:
        vol_support = low_1d[-1]
        vol_resistance = high_1d[-1]
    
    # Support/resistance arrays for 1d
    vol_support_arr = np.full_like(close_1d, vol_support)
    vol_resistance_arr = np.full_like(close_1d, vol_resistance)
    
    # Align 1d data to 6h
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d_values)
    vol_support_aligned = align_htf_to_ltf(prices, df_1d, vol_support_arr)
    vol_resistance_aligned = align_htf_to_ltf(prices, df_1d, vol_resistance_arr)
    
    # Calculate KAMA(10) on 6h closes
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    # Handle first 9 values where diff(10) is not available
    change_full = np.full(n, np.nan)
    volatility_full = np.full(n, np.nan)
    change_full[10:] = change
    volatility_full[10:] = volatility
    # For volatility, sum of absolute changes over 10 periods
    volatility_sum = np.full(n, np.nan)
    for i in range(10, n):
        volatility_sum[i] = np.sum(np.abs(np.diff(close[i-10:i])))
    
    # Avoid division by zero
    er = np.where(volatility_sum != 0, change_full / volatility_sum, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(kama[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_support_aligned[i]) or np.isnan(vol_resistance_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below support or RSI overbought
            if close[i] < vol_support_aligned[i] or rsi_1d_aligned[i] > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above resistance or RSI oversold
            if close[i] > vol_resistance_aligned[i] or rsi_1d_aligned[i] < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price near support, RSI oversold, above KAMA
            if (close[i] <= vol_support_aligned[i] * 1.01 and  # within 1% of support
                rsi_1d_aligned[i] < 30 and
                close[i] > kama[i]):
                position = 1
                signals[i] = 0.25
            # Short: price near resistance, RSI overbought, below KAMA
            elif (close[i] >= vol_resistance_aligned[i] * 0.99 and  # within 1% of resistance
                  rsi_1d_aligned[i] > 70 and
                  close[i] < kama[i]):
                position = -1
                signals[i] = -0.25
    
    return signals