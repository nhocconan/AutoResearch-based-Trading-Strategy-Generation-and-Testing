#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d EMA34 for trend filter (bull/bear regime).
- Entry: Long when price breaks above Camarilla R3 in bull regime with volume > 1.5 * 4h volume MA(20);
         Short when price breaks below Camarilla S3 in bear regime with volume > 1.5 * 4h volume MA(20).
- Exit: Opposite Camarilla breakout (below S3 for long, above R3 for short).
- Signal size: 0.25 discrete to balance capture and fee control.
- Camarilla levels provide precise intraday support/resistance; EMA34 adapts to trend; volume confirms conviction.
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
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get 1d data for EMA34 calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate 4h Camarilla levels (based on previous day's OHLC)
    # We need to calculate Camarilla for each 4h bar using prior 1d data
    # For simplicity, we'll use the prior completed 1d bar's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_r3 = np.zeros(len(df_1d))
    camarilla_s3 = np.zeros(len(df_1d))
    camarilla_r4 = np.zeros(len(df_1d))  # for exit
    camarilla_s4 = np.zeros(len(df_1d))  # for exit
    
    for i in range(len(df_1d)):
        h = high_1d[i]
        l = low_1d[i]
        c = close_1d[i]
        camarilla_r3[i] = c + (h - l) * 1.1 / 4
        camarilla_s3[i] = c - (h - l) * 1.1 / 4
        camarilla_r4[i] = c + (h - l) * 1.1 / 2
        camarilla_s4[i] = c - (h - l) * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate 4h volume MA(20) for confirmation
    volume_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 1)  # EMA34 needs 34, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_ma_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: 1.5x threshold
        vol_confirm = curr_volume > 1.5 * vol_ma_4h_aligned[i]
        
        # Trend filter: EMA34 direction
        bull_regime = curr_close > ema_34_aligned[i]
        bear_regime = curr_close < ema_34_aligned[i]
        
        if position == 0:
            # Check for entry signals
            # Long: price breaks above Camarilla R3 in bull regime with volume confirmation
            if curr_close > camarilla_r3_aligned[i] and bull_regime and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 in bear regime with volume confirmation
            elif curr_close < camarilla_s3_aligned[i] and bear_regime and vol_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: check exit (price breaks below Camarilla S3)
            if curr_close < camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: check exit (price breaks above Camarilla R3)
            if curr_close > camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0