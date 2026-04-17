#!/usr/bin/env python3
"""
Hypothesis: 12h strategy using 1d ATR-based volatility breakout with volume confirmation and 1d EMA50 trend filter.
- Long when price breaks above 1d close + 1.5 * ATR(14) + volume > 1.8x 20-period 12h volume MA + price above 1d EMA50
- Short when price breaks below 1d close - 1.5 * ATR(14) + volume > 1.8x 20-period 12h volume MA + price below 1d EMA50
- Fixed position size 0.25 to manage drawdown
- Uses volatility breakout structure (works in ranging and trending markets) + volume confirmation + trend filter
- Designed for 12h timeframe with strict entry conditions to limit trades to 50-150 total over 4 years
- ATR breakout captures expansion phases, effective in both accumulation (bull) and distribution (bear) phases
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
    
    # Get 12h data for volume MA
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    
    # Volume average (20-period) on 12h for confirmation
    volume_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for ATR, close, and EMA50 trend filter (HTF for structure)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d ATR(14)
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR-based breakout levels
    upper_break_1d = close_1d + 1.5 * atr_14_1d
    lower_break_1d = close_1d - 1.5 * atr_14_1d
    
    # Align all HTF indicators to primary timeframe (12h)
    volume_ma_20_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_20_12h)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    upper_break_aligned = align_htf_to_ltf(prices, df_1d, upper_break_1d)
    lower_break_aligned = align_htf_to_ltf(prices, df_1d, lower_break_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(volume_ma_20_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(upper_break_aligned[i]) or np.isnan(lower_break_aligned[i])):
            signals[i] = 0.0
            continue
        
        vol_ma = volume_ma_20_aligned[i]
        ema_50_val = ema_50_aligned[i]
        upper_break = upper_break_aligned[i]
        lower_break = lower_break_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for volatility breakouts with volume confirmation and 1d EMA50 trend filter
            # Long: price breaks above 1d upper ATR level + volume spike + price above 1d EMA50
            if price > upper_break and vol > 1.8 * vol_ma and price > ema_50_val:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below 1d lower ATR level + volume spike + price below 1d EMA50
            elif price < lower_break and vol > 1.8 * vol_ma and price < ema_50_val:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit on close below 1d EMA50 (trend change) or opposite breakout level
            if price < ema_50_val or price < lower_break:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit on close above 1d EMA50 (trend change) or opposite breakout level
            if price > ema_50_val or price > upper_break:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_ATRBreakout_VolumeSpike_1dEMA50"
timeframe = "12h"
leverage = 1.0