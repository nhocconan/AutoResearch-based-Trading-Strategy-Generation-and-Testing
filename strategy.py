#!/usr/bin/env python3
"""
6H Time-Weighted Average Price (TWAP) Pullback Strategy
- Calculates 6h TWAP as volume-weighted average of last 4 periods (~24h)
- Enters long when price pulls back to TWAP with bullish 1d candle and volume confirmation
- Enters short when price pulls back to TWAP with bearish 1d candle and volume confirmation
- Exits when price moves 1.5*ATR away from TWAP
- Designed for mean reversion in ranging markets with trend filter from higher timeframe
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_twap_pullback_1d_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === ATR (20) for stop loss ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # === TWAP (6-period VWAP ~ 24 hours) ===
    # Typical price * volume
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    # Sum of PV and volume over last 6 periods
    sum_pv = pd.Series(pv).rolling(window=6, min_periods=6).sum().values
    sum_vol = pd.Series(volume).rolling(window=6, min_periods=6).sum().values
    twap = sum_pv / (sum_vol + 1e-10)
    
    # === 1d trend filter (close vs open) ===
    df_1d = get_htf_data(prices, '1d')
    # Daily bullish/bearish: close > open
    daily_bullish = (df_1d['close'] > df_1d['open']).values
    daily_bullish_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish)
    # Convert boolean to float for alignment
    daily_bullish_aligned = daily_bullish_aligned.astype(float)
    
    # === Volume confirmation (20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if (np.isnan(twap[i]) or np.isnan(atr[i]) or 
            np.isnan(daily_bullish_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price moves 1.5*ATR above TWAP
            if close[i] > twap[i] + 1.5 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price moves 1.5*ATR below TWAP
            if close[i] < twap[i] - 1.5 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation
            if vol_ratio[i] < 1.1:
                signals[i] = 0.0
                continue
            
            # Calculate deviation from TWAP
            dev_pct = (close[i] - twap[i]) / twap[i]
            
            # Entry conditions: pullback to TWAP with 1d trend alignment
            # Long: price slightly below TWAP in bullish daily
            if close[i] <= twap[i] and daily_bullish_aligned[i] > 0.5 and dev_pct >= -0.005:
                # Allow small tolerance for entry
                position = 1
                signals[i] = 0.25
            # Short: price slightly above TWAP in bearish daily
            elif close[i] >= twap[i] and daily_bullish_aligned[i] < 0.5 and dev_pct <= 0.005:
                position = -1
                signals[i] = -0.25
    
    return signals