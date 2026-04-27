#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d VWAP deviation with 1w EMA50 trend filter and volume confirmation.
# Long when price closes below VWAP by 1.5x ATR(20) with 1w EMA50 uptrend and volume > 2x average.
# Short when price closes above VWAP by 1.5x ATR(20) with 1w EMA50 downtrend and volume > 2x average.
# Exit when price returns to VWAP.
# Uses VWAP deviation as mean reversion signal, targeting 15-30 trades per year.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema_period = 50
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= ema_period:
        ema_1w[ema_period - 1] = np.mean(close_1w[:ema_period])
        for i in range(ema_period, len(close_1w)):
            ema_1w[i] = (close_1w[i] * (2 / (ema_period + 1)) + 
                         ema_1w[i - 1] * (1 - (2 / (ema_period + 1))))
    
    # Calculate VWAP
    vwap = np.full(n, np.nan)
    cumulative_pv = np.cumsum(close * volume)
    cumulative_volume = np.cumsum(volume)
    vwap = cumulative_pv / cumulative_volume
    
    # Calculate ATR(20)
    atr_period = 20
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # First TR
    atr = np.full(n, np.nan)
    for i in range(atr_period - 1, n):
        atr[i] = np.mean(tr[i - atr_period + 1:i + 1])
    
    # Align 1w EMA to 1d timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume MA for confirmation (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need VWAP, ATR, EMA50, and volume MA20
    start_idx = max(atr_period, ema_period - 1, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(vwap[i]) or np.isnan(atr[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vwap_val = vwap[i]
        atr_val = atr[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require volume above average
        vol_filter = vol_now > 2.0 * vol_avg
        
        if position == 0:
            # Long: price below VWAP by 1.5x ATR with 1w EMA50 uptrend and volume filter
            if (price < vwap_val - 1.5 * atr_val and 
                price > ema_1w_aligned[i] and vol_filter):
                signals[i] = size
                position = 1
            # Short: price above VWAP by 1.5x ATR with 1w EMA50 downtrend and volume filter
            elif (price > vwap_val + 1.5 * atr_val and 
                  price < ema_1w_aligned[i] and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to VWAP
            if price >= vwap_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to VWAP
            if price <= vwap_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_VWAPDeviation_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0