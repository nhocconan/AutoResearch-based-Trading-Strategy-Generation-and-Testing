#!/usr/bin/env python3
# 4h_1d_ema_vwap_volatility_breakout_v1
# Strategy: 4h VWAP breakout with 1d EMA trend filter and volatility filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: VWAP acts as a dynamic support/resistance level. Breakouts above/below VWAP with
# institutional volume (volume > 1.5x 20-period average) and aligned with 1d EMA50 trend
# capture sustainable moves. Volatility filter (ATR ratio) avoids ranging markets.
# Designed for low trade frequency (<30/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_ema_vwap_volatility_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h VWAP calculation
    typical_price = (high + low + close) / 3
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = vwap_numerator / vwap_denominator
    
    # 4h ATR for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d volume average (20-period) for confirmation
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Align raw 1d volume for confirmation
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(vwap[i]) or np.isnan(atr[i]) or np.isnan(ema_50_1d_aligned[i]) or \
           np.isnan(vol_avg_20_1d_aligned[i]) or np.isnan(vol_1d_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volatility filter: avoid low volatility (range) markets
        # ATR ratio: current ATR vs 50-period average ATR
        if i >= 50:
            atr_ma = pd.Series(atr[:i+1]).rolling(window=50, min_periods=50).mean().iloc[-1]
            vol_filter = atr[i] > 0.8 * atr_ma  # Only trade when volatility is above 80% of average
        else:
            vol_filter = True
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        vol_confirm = vol_1d_aligned[i] > 1.5 * vol_avg_20_1d_aligned[i]
        
        # Trend filter: close vs 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Price relative to VWAP
        above_vwap = close[i] > vwap[i]
        below_vwap = close[i] < vwap[i]
        
        # Entry conditions
        # Long: Price crosses above VWAP AND uptrend AND volume confirmation AND volatility filter
        if above_vwap and uptrend and vol_confirm and vol_filter and position != 1:
            # Additional check: ensure we didn't just cross above VWAP in previous bar
            if i == 50 or close[i-1] <= vwap[i-1]:
                position = 1
                signals[i] = 0.25
        # Short: Price crosses below VWAP AND downtrend AND volume confirmation AND volatility filter
        elif below_vwap and downtrend and vol_confirm and vol_filter and position != -1:
            # Additional check: ensure we didn't just cross below VWAP in previous bar
            if i == 50 or close[i-1] >= vwap[i-1]:
                position = -1
                signals[i] = -0.25
        # Exit: Price crosses back through VWAP (mean reversion signal)
        elif position == 1 and below_vwap:
            position = 0
            signals[i] = 0.0
        elif position == -1 and above_vwap:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals