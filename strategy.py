#!/usr/bin/env python3
"""
Hypothesis: 12h strategy using Camarilla pivot (H3/L3) breakout with volume confirmation and 1d EMA50 trend filter.
- Long when price closes above 12h Camarilla H3 + volume > 1.5x 20-period 12h volume MA + price above 1d EMA50
- Short when price closes below 12h Camarilla L3 + volume > 1.5x 20-period 12h volume MA + price below 1d EMA50
- Fixed position size 0.25 to limit fee churn and manage drawdown
- ATR-based trailing stop (2.0x ATR) to lock in profits
- Designed for very low trade frequency (target: 50-150 trades over 4 years) to avoid fee drag
- Works in bull markets (buying breakouts above 1d EMA50) and bear markets (selling breakdowns below 1d EMA50)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 12h data for Camarilla pivots, volume confirmation, and ATR (primary timeframe)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Camarilla pivot levels on 12h (based on previous day's 12h OHLC)
    # For intraday timeframe, we use the previous 12h bar's range
    # H3 = C + (H-L)*1.1/4, L3 = C - (H-L)*1.1/4
    # We calculate these using rolling window of 2 (previous bar) to get the prior 12h bar's OHLC
    if len(high_12h) >= 2:
        # Shift to get previous bar's values
        prev_high = np.roll(high_12h, 1)
        prev_low = np.roll(low_12h, 1)
        prev_close = np.roll(close_12h, 1)
        # First value will be invalid due to roll, handle with min_periods logic
        prev_high[0] = np.nan
        prev_low[0] = np.nan
        prev_close[0] = np.nan
        
        rang = prev_high - prev_low
        h3 = prev_close + rang * 1.1 / 4
        l3 = prev_close - rang * 1.1 / 4
    else:
        h3 = np.full_like(close_12h, np.nan)
        l3 = np.full_like(close_12h, np.nan)
    
    # Volume average (20-period) on 12h for confirmation
    volume_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # ATR (10-period) on 12h for stoploss
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Align all indicators to 12h timeframe (primary)
    h3_aligned = align_htf_to_ltf(prices, df_12h, h3)
    l3_aligned = align_htf_to_ltf(prices, df_12h, l3)
    volume_ma_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_20)
    atr_aligned = align_htf_to_ltf(prices, df_12h, atr_10)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    atr_stop = 0.0
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(atr_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        h3_val = h3_aligned[i]
        l3_val = l3_aligned[i]
        vol_ma = volume_ma_aligned[i]
        atr_val = atr_aligned[i]
        ema_50_val = ema_50_1d_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation and 1d EMA50 trend filter
            # Long: price closes above H3 + volume spike + price above 1d EMA50
            if price > h3_val and vol > 1.5 * vol_ma and price > ema_50_val:
                signals[i] = 0.25
                position = 1
                entry_price = price
                atr_stop = entry_price - 2.0 * atr_val
            # Short: price closes below L3 + volume spike + price below 1d EMA50
            elif price < l3_val and vol > 1.5 * vol_ma and price < ema_50_val:
                signals[i] = -0.25
                position = -1
                entry_price = price
                atr_stop = entry_price + 2.0 * atr_val
        
        elif position == 1:
            # Check stoploss
            if price <= atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                # Trail stop: raise stop if price moves favorably
                atr_stop = max(atr_stop, price - 1.5 * atr_val)
        
        elif position == -1:
            # Check stoploss
            if price >= atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                # Trail stop: lower stop if price moves favorably
                atr_stop = min(atr_stop, price + 1.5 * atr_val)
    
    return signals

name = "12h_Camarilla_H3L3_1dEMA50_VolumeSpike_ATRTrail"
timeframe = "12h"
leverage = 1.0