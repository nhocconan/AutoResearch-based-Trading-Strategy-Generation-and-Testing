#!/usr/bin/env python3
"""
6h_HighLowPivot_Breakout_HTFTrend_V1
Hypothesis: 6h strategy using 12h HTF pivot levels (weekly high/low from prior 12h bar) as breakout triggers. Entry on break above weekly high (long) or below weekly low (short) with volume confirmation (>1.5x 20-period 6h volume MA) and HTF trend filter (12h EMA34). Uses ATR(14) trailing stop via signal=0 when price moves 2.5*ATR against position. Designed for low trade frequency (target 50-150 total trades over 4 years) to minimize fee drag and capture trending moves in both bull/bear markets via HTF trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (12h for pivot levels and EMA trend)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # === 12h HTF indicators ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Weekly high/low from prior completed 12h bar (using 2-bar lookback for weekly approx)
    # For 6h chart, weekly high/low approximated by max/min of last 2 12h bars (24h)
    weekly_high = pd.Series(high_12h).rolling(window=2, min_periods=2).max().shift(1).values
    weekly_low = pd.Series(low_12h).rolling(window=2, min_periods=2).min().shift(1).values
    
    # 12h EMA34 for trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 6h timeframe (completed-bar timing)
    weekly_high_aligned = align_htf_to_ltf(prices, df_12h, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_12h, weekly_low)
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # === 6h Indicators (primary timeframe) ===
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Volume MA (20-period) for confirmation
    vol_ma = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) for stoploss
    tr1 = pd.Series(high_6h - low_6h)
    tr2 = pd.Series(np.abs(high_6h - np.roll(close_6h, 1)))
    tr3 = pd.Series(np.abs(low_6h - np.roll(close_6h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) 
            or np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_6h[i]
        vol = volume_6h[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        
        # HTF trend filter
        uptrend = ema_34_12h_aligned[i] > ema_34_12h_aligned[i-1] if i > 0 else False
        downtrend = ema_34_12h_aligned[i] < ema_34_12h_aligned[i-1] if i > 0 else False
        
        if position == 0:
            # Long: break above weekly high + volume + uptrend
            if price > weekly_high_aligned[i] and vol_ok and uptrend:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: break below weekly low + volume + downtrend
            elif price < weekly_low_aligned[i] and vol_ok and downtrend:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit: break below weekly low or loss of volume/momentum
            elif price < weekly_low_aligned[i] or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit: break above weekly high or loss of volume/momentum
            elif price > weekly_high_aligned[i] or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_HighLowPivot_Breakout_HTFTrend_V1"
timeframe = "6h"
leverage = 1.0