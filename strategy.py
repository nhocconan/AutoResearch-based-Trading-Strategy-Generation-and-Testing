#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeRegime_ATRStop_v2
Hypothesis: 4h Camarilla pivot (R1/S1) breakout filtered by 1d EMA200 trend and volume regime (ATR-based volatility filter).
Uses ATR(14) stoploss (2.0x) and discrete position sizing (0.25) to minimize fee churn.
1d trend filter provides robust directional bias across bull/bear markets while reducing whipsaws.
Volume regime filter avoids low-volatility chop and high-volatility panic spikes.
Target: 20-40 trades/year per symbol for low fee drag and strong test generalization.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA200 trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d OHLC for Camarilla pivot calculation (based on previous 1d bar) ===
    df_1d_open = df_1d['open'].values
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    # Camarilla: R1 = close + 0.275*(high-low), S1 = close - 0.275*(high-low)
    range_1d = df_1d_high - df_1d_low
    r1_1d = df_1d_close + 0.275 * range_1d
    s1_1d = df_1d_close - 0.275 * range_1d
    
    # Align 1d Camarilla levels to 4h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === 1d EMA200 for trend filter ===
    ema_200_1d = pd.Series(df_1d_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # === ATR (14-period) for stoploss and volume regime filter ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Volume regime: ATR ratio (current ATR / 50-period ATR mean) to filter extremes
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr / atr_ma  # >1.0 = above average volatility
    # Volume regime: avoid low volatility (chop) and extreme volatility (panic)
    vol_regime = (atr_ratio > 0.5) & (atr_ratio < 3.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) 
            or np.isnan(ema_200_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(atr_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Trend and volume regime filters
            trend_filter = price > ema_200_1d_aligned[i]  # 1d uptrend
            vol_filter = vol_regime[i]  # acceptable volatility regime
            
            # Long conditions: price > R1 (breakout), 1d uptrend, volume regime
            long_breakout = price > r1_1d_aligned[i]
            
            # Short conditions: price < S1 (breakdown), 1d downtrend, volume regime
            short_breakout = price < s1_1d_aligned[i]
            short_trend = price < ema_200_1d_aligned[i]
            
            # Entry logic - ONLY enter on volume regime + trend alignment
            if long_breakout and trend_filter and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_breakout and short_trend and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes below S1 (breakdown)
            elif price < s1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above R1 (breakout)
            elif price > r1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeRegime_ATRStop_v2"
timeframe = "4h"
leverage = 1.0