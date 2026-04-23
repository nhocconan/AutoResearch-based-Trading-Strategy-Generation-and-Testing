#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla pivot breakout with 1d EMA34 trend filter and volume confirmation.
Long when price breaks above Camarilla R3 AND 1d EMA34 rising AND volume > 2.0x 20-period MA.
Short when price breaks below Camarilla S3 AND 1d EMA34 falling AND volume > 2.0x 20-period MA.
Exit when price touches opposite Camarilla level (S3 for long, R3 for short) or 1d EMA34 reverses.
Uses 1d HTF for trend filter to avoid counter-trend trades, volume spike for momentum confirmation.
Camarilla levels from 1d provide intraday structure proven to work on ETH/BTC. Volume filter reduces false breakouts.
Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
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
    
    # Calculate 6h Camarilla levels from prior 1d OHLC (updated only when 1d bar completes)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Camarilla levels: based on prior day's OHLC
    camarilla_R4 = np.full(n, np.nan)
    camarilla_R3 = np.full(n, np.nan)
    camarilla_S3 = np.full(n, np.nan)
    camarilla_S4 = np.full(n, np.nan)
    
    # Get prior day's close, high, low for each 6h bar
    for i in range(n):
        # Find the index of the most recent completed 1d bar
        # We use align_htf_to_ltf logic implicitly by using prior day's data
        # Simpler approach: for each 6h bar, use OHLC from the 1d bar that started at floor(6h timestamp to 1d)
        # But we'll compute Camarilla once per 1d and align
        pass
    
    # Instead: compute Camarilla for each 1d bar, then align to 6h
    # Typical Camarilla formula:
    # R4 = C + (H-L)*1.1/2
    # R3 = C + (H-L)*1.1/4
    # S3 = C - (H-L)*1.1/4
    # S4 = C - (H-L)*1.1/2
    # where C, H, L are from prior day
    
    # Calculate typical price for prior day
    typical_close_1d = df_1d['close'].values
    typical_high_1d = df_1d['high'].values
    typical_low_1d = df_1d['low'].values
    
    # Shift by 1 to use prior day's OHLC (avoid look-ahead)
    if len(typical_close_1d) < 2:
        return np.zeros(n)
    
    prior_close = np.roll(typical_close_1d, 1)
    prior_high = np.roll(typical_high_1d, 1)
    prior_low = np.roll(typical_low_1d, 1)
    # First bar has no prior day
    prior_close[0] = np.nan
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    
    camarilla_R4_1d = prior_close + (prior_high - prior_low) * 1.1 / 2
    camarilla_R3_1d = prior_close + (prior_high - prior_low) * 1.1 / 4
    camarilla_S3_1d = prior_close - (prior_high - prior_low) * 1.1 / 4
    camarilla_S4_1d = prior_close - (prior_high - prior_low) * 1.1 / 2
    
    # Align to 6h timeframe
    camarilla_R4 = align_htf_to_ltf(prices, df_1d, camarilla_R4_1d)
    camarilla_R3 = align_htf_to_ltf(prices, df_1d, camarilla_R3_1d)
    camarilla_S3 = align_htf_to_ltf(prices, df_1d, camarilla_S3_1d)
    camarilla_S4 = align_htf_to_ltf(prices, df_1d, camarilla_S4_1d)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(typical_close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA34 needs 34, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_R3[i]) or np.isnan(camarilla_S3[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        r3 = camarilla_R3[i]
        s3 = camarilla_S3[i]
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
        
        # Volume filter: 6h volume > 2.0x 20-period MA (higher threshold for fewer trades)
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
                # Long exit: price touches Camarilla S3 (opposite) OR EMA34 starts falling
                if price < s3 or (i >= start_idx + 1 and ema_val < ema_34_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches Camarilla R3 (opposite) OR EMA34 starts rising
                if price > r3 or (i >= start_idx + 1 and ema_val > ema_34_aligned[i-1]):
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