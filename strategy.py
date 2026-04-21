#!/usr/bin/env python3
"""
1d_Camarilla_Pivot_Breakout_WeeklyTrend_VolumeFilter
Hypothesis: Daily strategy using 1w Camarilla pivot levels (R1/S1) for breakout entries,
filtered by 1w EMA34 trend and volume confirmation. Enter long when price breaks above
1w R1 with 1w uptrend and above-average volume. Enter short when price breaks below
1w S1 with 1w downtrend and above-average volume. Exit on ATR(14) trailing stop (2.0*ATR)
or opposite level break. Target: 15-25 trades/year (~60-100 total over 4 years) to
minimize fee drag. Works in bull/bear via weekly trend alignment and volume filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for Camarilla pivots, EMA, ATR, volume)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # === 1w Camarilla Pivot Levels (R1, S1) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_range = (high_1w - low_1w) * 1.1 / 12.0
    r1_1w = close_1w + camarilla_range
    s1_1w = close_1w - camarilla_range
    
    # Align to daily timeframe (use previous completed weekly bar)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # === 1w EMA34 for HTF trend filter ===
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === 1w ATR (14-period) for stoploss ===
    tr1 = pd.Series(high_1w - low_1w)
    tr2 = pd.Series(np.abs(high_1w - np.roll(close_1w, 1)))
    tr3 = pd.Series(np.abs(low_1w - np.roll(close_1w, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1w = tr.rolling(window=14, min_periods=14).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # === 1w Volume average (20-period) for confirmation ===
    volume_1w = df_1w['volume'].values
    vol_ma_20 = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) 
            or np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr_1w_aligned[i])
            or np.isnan(vol_ma_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        
        if position == 0:
            # Long conditions: price > 1w R1, 1w uptrend, above-average volume
            long_breakout = price > r1_1w_aligned[i]
            long_trend = price > ema_34_1w_aligned[i]
            long_volume = prices['volume'].iloc[i] > vol_ma_20_aligned[i]
            
            # Short conditions: price < 1w S1, 1w downtrend, above-average volume
            short_breakout = price < s1_1w_aligned[i]
            short_trend = price < ema_34_1w_aligned[i]
            short_volume = prices['volume'].iloc[i] > vol_ma_20_aligned[i]
            
            # Entry logic
            if long_breakout and long_trend and long_volume:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_breakout and short_trend and short_volume:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes below 1w S1 (support broken)
            elif price < s1_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above 1w R1 (resistance broken)
            elif price > r1_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_Pivot_Breakout_WeeklyTrend_VolumeFilter"
timeframe = "1d"
leverage = 1.0