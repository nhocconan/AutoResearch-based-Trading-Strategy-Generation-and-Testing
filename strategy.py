#!/usr/bin/env python3
"""
4h_RSI20_Pullback_12hTrend_VolumeFilter
Hypothesis: Use RSI(20) pullback to 30/70 levels with 12h EMA50 trend filter and volume confirmation. RSI20 is more responsive than RSI14 for catching short-term reversals in 4h charts. Works in bull markets (buy pullbacks in uptrend) and bear markets (sell rallies in downtrend). Target 20-30 trades/year to minimize fee drag.
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # RSI(20) on 4h closes
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/20, adjust=False, min_periods=20).mean()
    avg_loss = loss.ewm(alpha=1/20, adjust=False, min_periods=20).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume confirmation: volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for RSI and EMA
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema50_12h_aligned[i]) or np.isnan(rsi_values[i]):
            signals[i] = 0.0
            continue
        
        ema_trend = ema50_12h_aligned[i]
        rsi_val = rsi_values[i]
        vol_ok = vol_filter[i]
        
        if position == 0:
            # Long: RSI pulls back to 30 in uptrend with volume
            if rsi_val <= 30 and ema_trend > 0 and vol_ok:
                signals[i] = size
                position = 1
            # Short: RSI rallies to 70 in downtrend with volume
            elif rsi_val >= 70 and ema_trend < 0 and vol_ok:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI reaches 70 or trend turns down
            if rsi_val >= 70 or ema_trend <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI reaches 30 or trend turns up
            if rsi_val <= 30 or ema_trend >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_RSI20_Pullback_12hTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0