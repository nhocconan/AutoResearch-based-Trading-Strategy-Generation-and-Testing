#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 reversal with 1d trend filter (EMA34) and volume spike confirmation.
Long when price breaks below S3 AND 1d EMA34 rising AND volume > 2.0x 24-period MA.
Short when price breaks above R3 AND 1d EMA34 falling AND volume > 2.0x 24-period MA.
Exit when price touches opposite Camarilla level (R2/S2) or 1d EMA34 reverses.
Uses 1d HTF for trend filter to align with major trend, Camarilla levels for mean reversion in ranges,
volume spike to confirm institutional interest at extremes. Target: 50-150 total trades over 4 years.
Camarilla R3/S3 represent strong support/resistance where reversals often occur in ranging markets.
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
    
    for i in range(1, n):
        # Use previous bar's OHLC to avoid look-ahead
        ph = high[i-1]
        pl = low[i-1]
        pc = close[i-1]
        rng = ph - pl
        
        camarilla_r3[i] = ph + rng * 1.1 / 4
        camarilla_s3[i] = pl - rng * 1.1 / 4
        camarilla_r2[i] = ph + rng * 1.1 / 6
        camarilla_s2[i] = pl - rng * 1.1 / 6
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6h volume MA (24-period) for spike filter
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(1, 34, 24)  # Camarilla (needs 1), EMA34, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_24[i])):
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
        vol_ma_val = vol_ma_24[i]
        
        # Calculate EMA34 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_34_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Volume filter: 6h volume > 2.0x 24-period MA (adaptive to volatility)
        vol_filter = volume[i] > 2.0 * vol_ma_val
        
        if position == 0:
            # Long: Break below S3 (mean reversion) AND 1d EMA34 rising AND volume filter
            if price < s3 and ema_rising and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Break above R3 (mean reversion) AND 1d EMA34 falling AND volume filter
            elif price > r3 and ema_falling and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches S2 (opposite Camarilla) OR 1d EMA34 starts falling
                if price > s2 or (i >= start_idx + 1 and ema_val < ema_34_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches R2 (opposite Camarilla) OR 1d EMA34 starts rising
                if price < r2 or (i >= start_idx + 1 and ema_val > ema_34_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Camarilla_R3S3_Reversal_1dEMA34_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0