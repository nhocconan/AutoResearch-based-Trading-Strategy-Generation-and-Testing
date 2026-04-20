#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian Breakout + 1d Volatility Filter + Volume Confirmation
# - Long when price breaks above Donchian(20) upper band (1d) + volume > 1.5x 20-period average + 1d volatility < 50-day percentile
# - Short when price breaks below Donchian(20) lower band (1d) + volume > 1.5x 20-period average + 1d volatility < 50-day percentile
# - Exit when price crosses back through Donchian(10) bands or volatility exceeds 80-day percentile
# - Uses daily Donchian channels for structure, volume for confirmation, and volatility regime filter
# - Designed for 12h timeframe with selective entries to avoid overtrading (target: 12-37 trades/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Donchian channels and volatility filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Donchian channels (20-period) for 1d
    high_max = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d volatility (ATR-like using true range)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first value
    atr_1d = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 50-day and 80-day volatility percentiles for regime filter
    vol_50d = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    vol_80d = pd.Series(atr_1d).rolling(window=80, min_periods=80).mean().values
    
    # Align Donchian levels and volatility filters to 12h timeframe
    high_max_aligned = align_htf_to_ltf(prices, df_1d, high_max)
    low_min_aligned = align_htf_to_ltf(prices, df_1d, low_min)
    vol_50d_aligned = align_htf_to_ltf(prices, df_1d, vol_50d)
    vol_80d_aligned = align_htf_to_ltf(prices, df_1d, vol_80d)
    
    # Calculate Donchian (10-period) for exit signals
    high_max_10 = pd.Series(high_1d).rolling(window=10, min_periods=10).max().values
    low_min_10 = pd.Series(low_1d).rolling(window=10, min_periods=10).min().values
    high_max_10_aligned = align_htf_to_ltf(prices, df_1d, high_max_10)
    low_min_10_aligned = align_htf_to_ltf(prices, df_1d, low_min_10)
    
    # Calculate 12h RSI for momentum filter
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume average (20-period)
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in indicators
        if np.isnan(high_max_aligned[i]) or np.isnan(low_min_aligned[i]) or \
           np.isnan(vol_50d_aligned[i]) or np.isnan(vol_80d_aligned[i]) or \
           np.isnan(rsi[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian(20) + volume spike + low volatility regime
            if price > high_max_aligned[i] and vol > 1.5 * vol_ma[i] and atr_1d[i] < vol_50d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian(20) + volume spike + low volatility regime
            elif price < low_min_aligned[i] and vol > 1.5 * vol_ma[i] and atr_1d[i] < vol_50d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below Donchian(10) or volatility expands
            if price < low_min_10_aligned[i] or atr_1d[i] > vol_80d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian(10) or volatility expands
            if price > high_max_10_aligned[i] or atr_1d[i] > vol_80d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_Volume_VolatilityFilter"
timeframe = "12h"
leverage = 1.0