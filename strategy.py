#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d EMA34 for trend direction (bullish when close > EMA34, bearish when close < EMA34).
- Entry: Long when price breaks above Camarilla R3 in 1d bull trend with volume > 2.0 * 4h volume MA(20); Short when price breaks below Camarilla S3 in 1d bear trend with volume > 2.0 * 4h volume MA(20).
- Exit: ATR-based trailing stop (2.5 * ATR(14)) or opposite Camarilla breakout.
- Signal size: 0.25 discrete to balance capture and fee control.
- Designed for BTC/ETH: Camarilla levels from 1d provide institutional support/resistance, EMA34 filter avoids counter-trend trades, volume spike confirms conviction, ATR stop manages risk.
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
    
    # Get 4h data for Camarilla and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels (R3, S3) from previous day
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    prev_close = df_4h['close'].shift(1).values
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    # Align Camarilla levels from 4h to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # Calculate 1d EMA34 for trend
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate 4h volume MA(20) for confirmation
    volume_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    # Calculate ATR(14) for trailing stop
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
    start_idx = max(34, 20, 14) + 1  # +1 for shift in Camarilla
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_4h_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: 2.0x threshold for spike
        vol_confirmed = curr_volume > 2.0 * vol_ma_4h_aligned[i]
        
        # Determine 1d EMA34 trend: bullish if close > EMA34, bearish if close < EMA34
        trend_bullish = close[i] > ema_34_aligned[i]
        trend_bearish = close[i] < ema_34_aligned[i]
        
        if position == 0:
            # Check for entry signals
            # Long: price breaks above Camarilla R3 in 1d bull trend with volume spike
            if curr_close > camarilla_r3_aligned[i] and trend_bullish and vol_confirmed:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_high
            # Short: price breaks below Camarilla S3 in 1d bear trend with volume spike
            elif curr_close < camarilla_s3_aligned[i] and trend_bearish and vol_confirmed:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_low
        elif position == 1:
            # Long position: update highest and check exit conditions
            highest_since_entry = max(highest_since_entry, curr_high)
            # Exit: ATR trailing stop or opposite Camarilla breakout
            if curr_low <= highest_since_entry - 2.5 * atr[i] or curr_close < camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: update lowest and check exit conditions
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit: ATR trailing stop or opposite Camarilla breakout
            if curr_high >= lowest_since_entry + 2.5 * atr[i] or curr_close > camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0