#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirm_v1
Hypothesis: Camarilla pivot R1/S1 breakout with 1d EMA34 trend filter and volume spike confirmation. Designed for low trade frequency (~20-40/year) to minimize fee drag. Uses 4h primary with 1d HTF for trend and volume context. Works in bull/bear via trend filter and volatility-based entry.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # === 1d trend filter: 34-period EMA ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 1d volume average (20-period) for spike detection ===
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d[np.isnan(vol_ma_1d)] = 1.0  # avoid division by zero
    vol_ratio_1d = volume_1d / vol_ma_1d
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === ATR for stoploss (14-period on 4h) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(atr_14[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        trend_1d = ema_34_1d_aligned[i]
        vol_spike = vol_ratio_1d_aligned[i]
        atr_val = atr_14[i]
        
        # Calculate Camarilla levels from previous day
        if i >= 1:
            # Get previous day's OHLC from 1d data
            prev_day_idx = len(df_1d) - 1  # align to current 4h bar's previous day
            # Find the 1d index that corresponds to the completed day before current 4h bar
            # Since we use align_htf_to_ltf, we can use the aligned arrays to get previous day's values
            # We'll approximate by using the 1d values from the prior aligned point
            # Simpler: use rolling window on 4h to get daily OHLC (acceptable approximation)
            pass  # We'll calculate Camarilla directly from 4h rolling window as proxy
        
        # Calculate Camarilla levels using 4h rolling window (24 periods = 1 day)
        # HLC of previous day: use rolling window of 24 periods (4h * 6 = 24h)
        if i >= 24:
            lookback = slice(i-24, i)  # previous 24 bars (1 day)
            ph = np.max(high[lookback])
            pl = np.min(low[lookback])
            pc = close[i-1]  # previous close
            
            # Camarilla R1, S1
            r1 = pc + (1.1/12) * (ph - pl)
            s1 = pc - (1.1/12) * (ph - pl)
        else:
            r1 = s1 = price_close  # fallback
        
        if position == 0:
            # Long: price breaks above R1 + volume spike > 2.0 + price above 1d EMA34
            if price_close > r1 and vol_spike > 2.0 and price_close > trend_1d:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
            # Short: price breaks below S1 + volume spike > 2.0 + price below 1d EMA34
            elif price_close < s1 and vol_spike > 2.0 and price_close < trend_1d:
                signals[i] = -0.25
                position = -1
                entry_price = price_close
        
        elif position != 0:
            # Stoploss: 2.0 * ATR against position
            if position == 1:
                if price_close < entry_price - 2.0 * atr_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price_close > entry_price + 2.0 * atr_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0