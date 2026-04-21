#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Fade_VolumeFilter
Hypothesis: Fade extreme Camarilla levels (R3/S3) on 6h timeframe with 1d trend filter.
Enter short at R3 when 1d EMA50 is below price (bearish bias), enter long at S3 when 1d EMA50 is above price (bullish bias).
Volume confirmation required to avoid false breaks in low liquidity.
Designed for low trade frequency (target: 12-25 trades/year) to minimize fee drag.
Works in bull/bear via 1d EMA50 trend alignment and volume spike filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA50 and Camarilla calculation)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Daily Camarilla Pivot Levels (R3, S3) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    camarilla_range = (high_1d - low_1d) * 1.1 / 4.0
    r3_1d = close_1d + camarilla_range
    s3_1d = close_1d - camarilla_range
    
    # Align to 1d timeframe (use previous completed daily bar)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # === Volume spike filter (20-period average) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) 
            or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]) 
            or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Volume confirmation: current volume > 20-period average
            vol_confirm = volume[i] > vol_ma[i]
            
            # Short conditions: price > daily R3, 1d EMA50 below price (bearish bias), volume spike
            short_breakout = price > r3_1d_aligned[i]
            short_trend = ema_50_1d_aligned[i] < price  # EMA below price = bearish
            
            # Long conditions: price < daily S3, 1d EMA50 above price (bullish bias), volume spike
            long_breakout = price < s3_1d_aligned[i]
            long_trend = ema_50_1d_aligned[i] > price  # EMA above price = bullish
            
            # Entry logic
            if short_breakout and short_trend and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
            elif long_breakout and long_trend and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above daily S3 (support reclaimed)
            elif price > s3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes below daily R3 (resistance reclaimed)
            elif price < r3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3_S3_Fade_VolumeFilter"
timeframe = "6h"
leverage = 1.0