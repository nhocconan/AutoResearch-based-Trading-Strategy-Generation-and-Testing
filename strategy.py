#!/usr/bin/env python3
"""
1d_KAMA_Regime_VolumeFilter_1wTrend
Hypothesis: Daily KAMA direction with 1-week EMA50 trend filter and volume confirmation (>1.3x 20-period MA).
Long when KAMA trending up, price above 1w EMA50, and volume > 1.3x average.
Short when KAMA trending down, price below 1w EMA50, and volume > 1.3x average.
Uses ATR-based stop (2.5x) and minimum holding period of 2 days to reduce churn.
Designed for low trade frequency (~10-20/year) to work in both bull and bear markets via 1w trend alignment.
KAMA adapts to market noise, reducing whipsaw in sideways markets while capturing major trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for EMA trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1w EMA50 for trend regime ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === Daily ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === Volume confirmation (1.3x 20-period MA) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Daily KAMA (10, 2, 30) ===
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)[:len(change)]  # sum of abs changes over 10 periods
    # Pad arrays to match length
    change_padded = np.concatenate([np.full(10, np.nan), change])
    volatility_padded = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility_padded != 0, change_padded / volatility_padded, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(50, n):  # warmup for indicators
        # Skip if indicators not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(kama[i]) or i < 10):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        ema_50_1w_val = ema_50_1w_aligned[i]
        vol_avg = vol_ma[i]
        kama_val = kama[i]
        kama_prev = kama[i-1]
        
        # Volume confirmation: current volume > 1.3x average
        volume_confirm = volume_now > 1.3 * vol_avg
        # KAMA direction: rising if current > previous
        kama_rising = kama_val > kama_prev
        kama_falling = kama_val < kama_prev
        
        if position == 0:
            # Long: KAMA rising, price above 1w EMA50, volume confirm
            long_condition = kama_rising and (price > ema_50_1w_val) and volume_confirm
            # Short: KAMA falling, price below 1w EMA50, volume confirm
            short_condition = kama_falling and (price < ema_50_1w_val) and volume_confirm
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
                bars_since_entry = 0
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
                bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Minimum holding period of 2 days to reduce churn
            if bars_since_entry < 2:
                signals[i] = 0.25 if position == 1 else -0.25
                continue
            
            # Check stoploss (2.5x ATR)
            if position == 1:
                if price < entry_price - 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Trend reversal exit (price below 1w EMA50)
                elif price < ema_50_1w_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > entry_price + 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Trend reversal exit (price above 1w EMA50)
                elif price > ema_50_1w_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Regime_VolumeFilter_1wTrend"
timeframe = "1d"
leverage = 1.0