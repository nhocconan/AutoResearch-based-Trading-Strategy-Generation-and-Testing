#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume confirmation.
Long when price breaks above Camarilla R1 AND 1d EMA34 rising AND 12h volume > 1.5x 20-period MA.
Short when price breaks below Camarilla S1 AND 1d EMA34 falling AND 12h volume > 1.5x 20-period MA.
Exit when price touches opposite Camarilla level (S1 for long, R1 for short) or 1d EMA34 reverses.
Uses 1d HTF for trend filter to avoid counter-trend trades, volume confirmation for momentum.
Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
Camarilla levels provide structure, 1d EMA34 filters major trend, volume filter avoids low-momentum breakouts.
Designed to work in bull (trend filters) and bear (volume spikes on breakdowns) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Camarilla levels (R1, S1) based on previous bar's range
    camarilla_r1 = np.full(n, np.nan)
    camarilla_s1 = np.full(n, np.nan)
    camarilla_r2 = np.full(n, np.nan)  # for exit reference
    camarilla_s2 = np.full(n, np.nan)  # for exit reference
    
    # Pivot point = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    # R2 = C + (H - L) * 1.1 / 6
    # S2 = C - (H - L) * 1.1 / 6
    for i in range(1, n):
        ph = high[i-1]
        pl = low[i-1]
        pc = close[i-1]
        rng = ph - pl
        if rng > 0:
            pivot = (ph + pl + pc) / 3.0
            camarilla_r1[i] = pivot + rng * 1.1 / 12.0
            camarilla_s1[i] = pivot - rng * 1.1 / 12.0
            camarilla_r2[i] = pivot + rng * 1.1 / 6.0
            camarilla_s2[i] = pivot - rng * 1.1 / 6.0
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 12h volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(1, 34, 20)  # Camarilla needs previous bar, EMA34, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        r1 = camarilla_r1[i]
        s1 = camarilla_s1[i]
        r2 = camarilla_r2[i]
        s2 = camarilla_s2[i]
        ema_val = ema_34_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Calculate EMA34 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_34_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Volume filter: 12h volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_val
        
        if position == 0:
            # Long: Break above Camarilla R1 AND EMA34 rising AND volume filter
            if price > r1 and ema_rising and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Break below Camarilla S1 AND EMA34 falling AND volume filter
            elif price < s1 and ema_falling and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches Camarilla S1 (opposite level) OR EMA34 starts falling
                if price <= s1 or (i >= start_idx + 1 and ema_val < ema_34_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches Camarilla R1 (opposite level) OR EMA34 starts rising
                if price >= r1 or (i >= start_idx + 1 and ema_val > ema_34_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0