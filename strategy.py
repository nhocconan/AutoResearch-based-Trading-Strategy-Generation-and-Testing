#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h EMA34 for trend filter (defines bull/bear regime).
- Entry: Long when price breaks above Camarilla R3 in bull regime with volume > 2.0 * 4h volume MA(20);
         Short when price breaks below Camarilla S3 in bear regime with volume > 2.0 * 4h volume MA(20).
- Exit: ATR trailing stop (2.5 * ATR(14)) or opposite Camarilla breakout.
- Signal size: 0.25 discrete to balance capture and fee control.
- Camarilla levels provide institutional pivot points; EMA34 adapts to trend; volume spike confirms conviction.
- Works in bull (breakouts with trend) and bear (strong moves after regime shifts).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla calculation and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 12h data for EMA34 calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA34
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 4h Camarilla levels (based on previous day's OHLC)
    # We'll use 4h data but calculate based on daily pivot concept adapted to 4h
    # For simplicity, we use 20-period high/low/close as proxy for daily
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate pivot point and Camarilla levels using 20-period lookback
    # This approximates the daily Camarilla calculation on 4h data
    hh20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    ll20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    cc20 = pd.Series(close_4h).rolling(window=20, min_periods=20).mean().values
    
    pivot = (hh20 + ll20 + cc20) / 3
    range_20 = hh20 - ll20
    
    # Camarilla levels
    r3 = pivot + (range_20 * 1.1 / 4)
    s3 = pivot - (range_20 * 1.1 / 4)
    
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3)
    
    # Calculate 4h volume MA(20) for confirmation
    volume_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    # Calculate 4h ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0
    lowest_since_entry = 0
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 20, 14, 1)  # EMA34 needs 34, Camarilla needs 20, volume MA needs 20, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ma_4h_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: 2.0x threshold (tight to reduce trades)
        vol_confirm = curr_volume > 2.0 * vol_ma_4h_aligned[i]
        
        # Trend filter: 12h EMA34 direction
        bull_regime = close[i] > ema_12h_aligned[i]
        bear_regime = close[i] < ema_12h_aligned[i]
        
        if position == 0:
            # Check for entry signals
            # Long: price breaks above Camarilla R3 in bull regime with volume confirmation
            if curr_close > r3_aligned[i] and bull_regime and vol_confirm:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_high
            # Short: price breaks below Camarilla S3 in bear regime with volume confirmation
            elif curr_close < s3_aligned[i] and bear_regime and vol_confirm:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_low
        elif position == 1:
            # Long position: update highest and check exit conditions
            highest_since_entry = max(highest_since_entry, curr_high)
            # Exit: ATR trailing stop or opposite breakout (below Camarilla S3)
            if curr_low <= highest_since_entry - 2.5 * atr[i] or curr_close < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: update lowest and check exit conditions
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit: ATR trailing stop or opposite breakout (above Camarilla R3)
            if curr_high >= lowest_since_entry + 2.5 * atr[i] or curr_close > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_12hEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0