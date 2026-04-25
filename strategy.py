#!/usr/bin/env python3
"""
12h_KAMA_Direction_RSI_Filter_VolumeSpike
Hypothesis: On 12h timeframe, Kaufman Adaptive Moving Average (KAMA) trend direction combined with RSI(14) > 50 for longs and < 50 for shorts, plus volume spike (>2.0x 20-bar average) captures adaptive trend momentum with controlled trade frequency. KAMA adjusts to market noise, reducing whipsaws in ranging markets while maintaining trend-following capability. Volume confirmation ensures institutional participation. Discrete sizing (0.25) minimizes fee churn. Works in bull markets via long entries and bear markets via short entries. Uses 1d HTF for volume average to avoid look-ahead and ensure proper alignment.
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
    
    # Get 1d data for HTF volume average (primary HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate KAMA on 1d close for trend direction
    # KAMA parameters: ER period=10, Fast SC=2/(2+1)=0.6667, Slow SC=2/(30+1)=0.0645
    close_1d_series = pd.Series(close_1d)
    change = abs(close_1d_series.diff(10))
    volatility = close_1d_series.diff(1).abs().rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    sc = (er * (0.6667 - 0.0645) + 0.0645) ** 2
    sc = sc.fillna(0.0645)  # default to slow SC when ER is NaN
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close_1d[i] - kama[i-1])
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate RSI(14) on 1d close
    delta = close_1d_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)  # neutral when undefined
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi.values)
    
    # Calculate volume average (20-period) on 1d for volume spike filter
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(34, 20)  # KAMA needs ~34 bars to stabilize, vol MA 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        vol_ma_val = vol_ma_aligned[i]
        vol_val = volume[i]
        close_val = close[i]
        
        # Volume spike condition: current volume > 2.0x 20-period average
        volume_spike = vol_val > 2.0 * vol_ma_val
        
        if position == 0:
            # Look for entry signals: KAMA direction with RSI filter and volume spike
            # Long: price above KAMA (uptrend) AND RSI > 50 (bullish momentum) AND volume spike
            long_signal = (close_val > kama_val) and (rsi_val > 50) and volume_spike
            # Short: price below KAMA (downtrend) AND RSI < 50 (bearish momentum) AND volume spike
            short_signal = (close_val < kama_val) and (rsi_val < 50) and volume_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Trend change: price crosses below KAMA
            if close_val < kama_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Trend change: price crosses above KAMA
            if close_val > kama_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "12h_KAMA_Direction_RSI_Filter_VolumeSpike"
timeframe = "12h"
leverage = 1.0