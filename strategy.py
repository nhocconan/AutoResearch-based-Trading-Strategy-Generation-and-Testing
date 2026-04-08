#!/usr/bin/env python3
# 1h_4h1d_trend_regime_v1
# Hypothesis: 1h EMA(21) pullback strategy with 4h EMA(50) trend filter and 1d chop regime filter.
# Long: price > 4h EMA50 AND price pulls back to 1h EMA21 with volume > 1.5x average AND 1d chop < 61.8 (trending)
# Short: price < 4h EMA50 AND price pulls back to 1h EMA21 with volume > 1.5x average AND 1d chop < 61.8 (trending)
# Exit: price crosses 1h EMA21 in opposite direction OR 1d chop > 61.8 (ranging)
# Session filter: 08-20 UTC to avoid low liquidity periods
# Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag.
# Uses 1h primary timeframe with 4h/1d HTF for regime filters.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_trend_regime_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1h EMA(21) for entry/exit
    close_s = pd.Series(close)
    ema_21 = close_s.ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend direction
    close_4h = pd.Series(df_4h['close'].values)
    ema_50_4h = close_4h.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Chopiness Index on 1d data (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    chop_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        # True Range calculation
        tr1 = high_1d[i] - low_1d[i]
        tr2 = abs(high_1d[i] - close_1d[i-1])
        tr3 = abs(low_1d[i] - close_1d[i-1])
        tr = max(tr1, tr2, tr3)
        
        # Sum of TR for last 14 periods
        atr_sum = 0
        for j in range(i-13, i+1):
            tr1_j = high_1d[j] - low_1d[j]
            tr2_j = abs(high_1d[j] - close_1d[j-1])
            tr3_j = abs(low_1d[j] - close_1d[j-1])
            tr_j = max(tr1_j, tr2_j, tr3_j)
            atr_sum += tr_j
        
        atr = atr_sum / 14
        max_high = np.max(high_1d[i-13:i+1])
        min_low = np.min(low_1d[i-13:i+1])
        
        if max_high != min_low:
            chop_1d[i] = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
        else:
            chop_1d[i] = 50  # neutral when no range
    
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate volume ratio (current vs 20-period average)
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    vol_ratio = np.where(vol_sma > 0, volume / vol_sma, 0)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # open_time is already datetime64[ms]
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        vol_r = vol_ratio[i]
        ch = chop_1d_aligned[i]
        price = close[i]
        ema21 = ema_21[i]
        trend = ema_50_4h_aligned[i]
        
        if np.isnan(vol_r) or np.isnan(ch) or np.isnan(ema21) or np.isnan(trend):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            if price < ema21 or ch > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            if price > ema21 or ch > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Long entry: price above 4h EMA50 AND pullback to 1h EMA21 with volume confirmation
            if price > trend and price <= ema21 * 1.005 and vol_r > 1.5 and ch < 61.8:
                position = 1
                signals[i] = 0.20
            # Short entry: price below 4h EMA50 AND pullback to 1h EMA21 with volume confirmation
            elif price < trend and price >= ema21 * 0.995 and vol_r > 1.5 and ch < 61.8:
                position = -1
                signals[i] = -0.20
    
    return signals