# HYPOTHESIS: Combines 1-day RSI for mean-reversion signals with 12-hour ADX trend filter and volume confirmation. 
# RSI < 30 oversold + ADX > 25 trending = long setup. RSI > 70 overbought + ADX > 25 = short setup.
# Works in bull/bear: RSI identifies extremes, ADX avoids ranging markets, volume confirms momentum.
# Target: 20-40 trades/year (80-160 total over 4 years) with size 0.25.

#!/usr/bin/env python3
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
    
    # Get 1-day data for RSI calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 14-period RSI on daily closes
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(gain, np.nan, dtype=float)
    avg_loss = np.full_like(loss, np.nan, dtype=float)
    
    if len(gain) >= 14:
        # Initial average
        avg_gain[13] = np.mean(gain[1:14])
        avg_loss[13] = np.mean(loss[1:14])
        # Wilder smoothing
        for i in range(14, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_14_1d = 100 - (100 / (1 + rs))
    
    # Get 12-hour data for ADX calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate True Range and Directional Movement
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR
    
    plus_dm = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h), 
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    minus_dm = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)), 
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smooth TR, +DM, -DM with Wilder's smoothing (14-period)
    atr_12h = np.full_like(tr, np.nan, dtype=float)
    plus_di_12h = np.full_like(tr, np.nan, dtype=float)
    minus_di_12h = np.full_like(tr, np.nan, dtype=float)
    
    if len(tr) >= 14:
        atr_12h[13] = np.mean(tr[1:14])
        plus_dm_sum = np.mean(plus_dm[1:14])
        minus_dm_sum = np.mean(minus_dm[1:14])
        
        for i in range(14, len(tr)):
            atr_12h[i] = (atr_12h[i-1] * 13 + tr[i]) / 14
            plus_dm_sum = (plus_dm_sum * 13 + plus_dm[i]) / 14
            minus_dm_sum = (minus_dm_sum * 13 + minus_dm[i]) / 14
            plus_di_12h[i] = 100 * plus_dm_sum / atr_12h[i] if atr_12h[i] != 0 else 0
            minus_di_12h[i] = 100 * minus_dm_sum / atr_12h[i] if atr_12h[i] != 0 else 0
    
    # Calculate DX and ADX
    dx = np.full_like(tr, np.nan, dtype=float)
    adx_12h = np.full_like(tr, np.nan, dtype=float)
    
    if len(tr) >= 28:  # Need 14 for DI + 14 for ADX
        for i in range(27, len(tr)):
            di_sum = plus_di_12h[i] + minus_di_12h[i]
            if di_sum != 0:
                dx[i] = 100 * np.abs(plus_di_12h[i] - minus_di_12h[i]) / di_sum
        
        if len(dx) >= 28:
            adx_12h[27] = np.mean(dx[14:28])
            for i in range(28, len(dx)):
                adx_12h[i] = (adx_12h[i-1] * 13 + dx[i]) / 14
    
    # Calculate 6-period volume average for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 6
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    # Align indicators to 6h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(30, vol_period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(rsi_aligned[i]) or np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume filter: at least 1.3x average volume
        vol_filter = vol_ratio > 1.3
        
        if position == 0:
            # Long: RSI oversold (<30), ADX trending (>25), volume confirmation
            if rsi_aligned[i] < 30 and adx_aligned[i] > 25 and vol_filter:
                signals[i] = size
                position = 1
            # Short: RSI overbought (>70), ADX trending (>25), volume confirmation
            elif rsi_aligned[i] > 70 and adx_aligned[i] > 25 and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: RSI returns to neutral (>50) or volatility spike
            if rsi_aligned[i] > 50 or vol_ratio > 2.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: RSI returns to neutral (<50) or volatility spike
            if rsi_aligned[i] < 50 or vol_ratio > 2.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_RSI_ADX_Volume_Filter"
timeframe = "6h"
leverage = 1.0