#!/usr/bin/env python3
# 6H_Adaptive_Kelly_RSIVolume
# Hypothesis: Adaptive Kelly sizing based on RSI extremes and volume spikes on 6h timeframe.
# Uses 1d trend filter and weekly volatility regime to adapt position size.
# Long when RSI < 30 and volume > 2x average in 1d uptrend (low vol regime).
# Short when RSI > 70 and volume > 2x average in 1d downtrend (low vol regime).
# Kelly fraction scales with signal strength (RSI extremity and volume ratio).
# Designed for low trade frequency (~20-40/year) with controlled risk in bull/bear markets.

name = "6H_Adaptive_Kelly_RSIVolume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6h indicators
    close_s = pd.Series(close)
    volume_s = pd.Series(volume)
    
    # RSI(14)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume average (20-period)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Daily trend and volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily EMA50 trend
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_uptrend = close_1d > ema50_1d
    daily_downtrend = close_1d < ema50_1d
    
    # Weekly volatility regime (using 1d data)
    # ATR(5) / ATR(20) ratio - low when < 0.8
    tr_1d = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
    )
    atr5_1d = pd.Series(tr_1d).ewm(span=5, adjust=False, min_periods=5).mean()
    atr20_1d = pd.Series(tr_1d).ewm(span=20, adjust=False, min_periods=20).mean()
    vol_ratio_1d = (atr5_1d / (atr20_1d + 1e-10)).values
    vol_ratio_1d = np.concatenate([np.full(1, np.nan), vol_ratio_1d])  # align with close_1d
    low_vol_regime = vol_ratio_1d < 0.8
    
    # Align daily indicators to 6h
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend.astype(float))
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend.astype(float))
    low_vol_aligned = align_htf_to_ltf(prices, df_1d, low_vol_regime.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi[i]) or np.isnan(vol_ma[i]) or
            np.isnan(daily_uptrend_aligned[i]) or np.isnan(daily_downtrend_aligned[i]) or
            np.isnan(low_vol_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_spike = vol_ratio > 2.0
        low_vol = low_vol_aligned[i] > 0.5
        
        daily_up = daily_uptrend_aligned[i] > 0.5
        daily_down = daily_downtrend_aligned[i] > 0.5
        
        rsi_val = rsi[i]
        # Calculate signal strength based on RSI extremity and volume
        if rsi_val < 30:
            rsi_strength = (30 - rsi_val) / 30  # 0 to 1
        elif rsi_val > 70:
            rsi_strength = (rsi_val - 70) / 30  # 0 to 1
        else:
            rsi_strength = 0
        
        # Kelly fraction approximation: edge * volatility scaling
        # Base size 0.25, scaled by signal strength (max 0.5)
        base_size = 0.25
        if rsi_strength > 0 and volume_spike and low_vol:
            signal_size = base_size * (0.5 + 0.5 * rsi_strength)  # 0.25 to 0.375
        else:
            signal_size = 0
        
        if position == 0:
            # Enter long: oversold + volume spike + 1d uptrend + low vol regime
            if rsi_val < 30 and volume_spike and daily_up and low_vol:
                signals[i] = signal_size
                position = 1
            # Enter short: overbought + volume spike + 1d downtrend + low vol regime
            elif rsi_val > 70 and volume_spike and daily_down and low_vol:
                signals[i] = -signal_size
                position = -1
        
        elif position == 1:
            # Exit when RSI returns to neutral or trend changes
            if rsi_val > 50 or not daily_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = signal_size
        
        elif position == -1:
            # Exit when RSI returns to neutral or trend changes
            if rsi_val < 50 or not daily_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -signal_size
    
    return signals