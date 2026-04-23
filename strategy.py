#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
Long when price breaks above Camarilla R3 AND 1d EMA34 rising AND volume > 2x 20-period MA.
Short when price breaks below Camarilla S3 AND 1d EMA34 falling AND volume > 2x 20-period MA.
Exit when price touches opposite Camarilla level (R2/S2) or 1d EMA34 reverses.
Uses 1d HTF for trend filter to avoid counter-trend trades in bear markets, volume spike for confirmation.
Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
Camarilla provides intraday structure, 1d EMA34 filters major trend, volume confirms breakout strength.
Designed to work in both bull and bear markets by following higher timeframe trend and requiring volume confirmation.
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
    
    # Calculate 6h Camarilla levels (based on previous bar's OHLC)
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_r2 = np.full(n, np.nan)
    camarilla_s2 = np.full(n, np.nan)
    camarilla_r4 = np.full(n, np.nan)
    camarilla_s4 = np.full(n, np.nan)
    camarilla_r1 = np.full(n, np.nan)
    camarilla_s1 = np.full(n, np.nan)
    camarilla_ph = np.full(n, np.nan)  # pivot high
    camarilla_pl = np.full(n, np.nan)  # pivot low
    camarilla_pc = np.full(n, np.nan)  # pivot close
    
    for i in range(1, n):
        # Use previous bar's OHLC to avoid look-ahead
        ph = high[i-1]
        pl = low[i-1]
        pc = close[i-1]
        
        camarilla_ph[i] = ph
        camarilla_pl[i] = pl
        camarilla_pc[i] = pc
        
        # Camarilla levels calculation
        rang = ph - pl
        camarilla_r4[i] = pc + rang * 1.1 / 2
        camarilla_r3[i] = pc + rang * 1.1 / 4
        camarilla_r2[i] = pc + rang * 1.1 / 6
        camarilla_r1[i] = pc + rang * 1.1 / 12
        camarilla_s1[i] = pc - rang * 1.1 / 12
        camarilla_s2[i] = pc - rang * 1.1 / 6
        camarilla_s3[i] = pc - rang * 1.1 / 4
        camarilla_s4[i] = pc - rang * 1.1 / 2
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(1, 34, 20)  # Camarilla (needs 1), EMA34, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(camarilla_r2[i]) or np.isnan(camarilla_s2[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        r3 = camarilla_r3[i]
        s3 = camarilla_s3[i]
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
        
        # Volume filter: 6h volume > 2x 20-period MA (adaptive to volatility)
        vol_filter = volume[i] > 2.0 * vol_ma_val
        
        if position == 0:
            # Long: Break above Camarilla R3 AND EMA34 rising AND volume filter
            if price > r3 and ema_rising and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Break below Camarilla S3 AND EMA34 falling AND volume filter
            elif price < s3 and ema_falling and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches Camarilla R2 (profit target) OR S2 (stop) OR EMA34 starts falling
                if price >= r2 or price <= s2 or (i >= start_idx + 1 and ema_val < ema_34_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches Camarilla S2 (profit target) OR R2 (stop) OR EMA34 starts rising
                if price <= s2 or price >= r2 or (i >= start_idx + 1 and ema_val > ema_34_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0