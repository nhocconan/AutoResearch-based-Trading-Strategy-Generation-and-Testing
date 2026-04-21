#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Fade_Reverse_v1
Hypothesis: Price tends to reverse at Camarilla R3/S3 levels with 1d trend filter on 6h timeframe.
In uptrend (1d close > 1d EMA34), fade moves toward R3/S3 for mean reversion.
In downtrend (1d close < 1d EMA34), fade moves toward R3/S3 for mean reversion.
Uses volume confirmation to avoid false signals. Works in bull/bear markets by fading extremes within the trend.
Target: 12-30 trades/year per symbol (50-120 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data once for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla levels calculation
    range_ = prev_high - prev_low
    # R3, S3 levels (most significant for reversals)
    r3 = prev_close + range_ * 1.1 / 4
    s3 = prev_close - range_ * 1.1 / 4
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: 24-period average (4 days on 6h)
    vol_ma = prices['volume'].rolling(window=24, min_periods=24).mean().values
    
    # ATR for position sizing and stop distance
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation
        volume_ok = volume > 1.5 * vol_ma[i]
        
        # 1d trend filter
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price near S3 in uptrend with volume (fade down move)
            if uptrend and volume_ok:
                if price <= s3_aligned[i] + 0.2 * atr[i]:
                    signals[i] = 0.25
                    position = 1
            # Short: price near R3 in downtrend with volume (fade up move)
            elif downtrend and volume_ok:
                if price >= r3_aligned[i] - 0.2 * atr[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: price reaches midpoint or stoploss
            midpoint = (r3_aligned[i] + s3_aligned[i]) / 2
            if price >= midpoint or price < s3_aligned[i] - 1.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price reaches midpoint or stoploss
            midpoint = (r3_aligned[i] + s3_aligned[i]) / 2
            if price <= midpoint or price > r3_aligned[i] + 1.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S3_Fade_Reverse_v1"
timeframe = "6h"
leverage = 1.0