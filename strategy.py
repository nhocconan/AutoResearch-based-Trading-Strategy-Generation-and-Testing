#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot reversal with 1d trend filter and volume spike
# Uses 1d EMA34 for trend direction (long when price > EMA34, short when price < EMA34)
# and Camarilla pivot levels (R1/S1) from 1d OHLC for reversal entries.
# Volume > 2.0x 24-period average confirms institutional interest at pivot levels.
# Camarilla reversals work well in ranging markets while trend filter avoids counter-trend trades.
# Target: 20-30 trades/year to minimize fee decay while capturing high-probability reversals.
# Focus on BTC/ETH as primary assets with proven Camarilla edge from DB.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_for_pivot = df_1d['close'].values
    
    # Camarilla: Range = (H - L), then levels = C ± (Range * multiplier)
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_multiplier = 1.1 / 12
    r1 = close_1d_for_pivot + (high_1d - low_1d) * camarilla_multiplier
    s1 = close_1d_for_pivot - (high_1d - low_1d) * camarilla_multiplier
    
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 24-period average volume for spike detection (4h bars in 1d = 6, so 24 = 4d)
    vol_ma = np.full(n, np.nan)
    vol_period = 24
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period
    start_idx = max(vol_period, 1)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Determine trend from 1d EMA34
        uptrend = price > ema_34_1d_aligned[i]
        downtrend = price < ema_34_1d_aligned[i]
        
        # Volume confirmation: spike > 2.0x average
        volume_confirmation = vol_ratio > 2.0
        
        if position == 0:
            # Long reversal at S1: price bounces up from support in uptrend
            if uptrend and price <= s1_aligned[i] * 1.001 and volume_confirmation:  # 0.1% buffer
                signals[i] = size
                position = 1
            # Short reversal at R1: price rejects down from resistance in downtrend
            elif downtrend and price >= r1_aligned[i] * 0.999 and volume_confirmation:  # 0.1% buffer
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price reaches R1 or breaks below EMA34
            if price >= r1_aligned[i] * 0.999 or price < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price reaches S1 or breaks above EMA34
            if price <= s1_aligned[i] * 1.001 or price > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1S1_1dEMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0