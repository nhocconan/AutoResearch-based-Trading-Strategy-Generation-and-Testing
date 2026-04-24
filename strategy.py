#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA(34) trend filter and volume spike confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d EMA(34) for trend filter (defines bull/bear regime).
- Entry: Long when price breaks above Camarilla R3 in bull regime with volume > 2.0 * 6h volume MA(20);
         Short when price breaks below Camarilla S3 in bear regime with volume > 2.0 * 6h volume MA(20).
- Exit: Price crosses below Camarilla R2 for long or above Camarilla S2 for short.
- Signal size: 0.25 discrete to balance capture and fee control.
- Camarilla pivot levels from 1d provide strong intraday support/resistance; breakouts with volume confirm institutional participation.
- Works in bull (buying R3 breakouts in uptrend) and bear (selling S3 breakdowns in downtrend).
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
    
    # Get 6h data for volume MA calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivot and EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34)
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 6h volume MA(20) for confirmation
    volume_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    # Calculate 1d Camarilla pivot levels (using previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    R3 = pivot + (range_1d * 1.1 / 4.0)
    S3 = pivot - (range_1d * 1.1 / 4.0)
    R2 = pivot + (range_1d * 1.1 / 6.0)
    S2 = pivot - (range_1d * 1.1 / 6.0)
    
    # Align Camarilla levels to 6h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA needs 34, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma_6h_aligned[i]) or 
            np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(R2_aligned[i]) or np.isnan(S2_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: 2.0x threshold (strict to reduce trades)
        vol_confirm = curr_volume > 2.0 * vol_ma_6h_aligned[i]
        
        # Trend filter: price relative to 1d EMA
        bull_regime = curr_close > ema_1d_aligned[i]
        bear_regime = curr_close < ema_1d_aligned[i]
        
        if position == 0:
            # Check for entry signals
            # Long: price breaks above Camarilla R3 in bull regime with volume confirmation
            if curr_high > R3_aligned[i] and bull_regime and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 in bear regime with volume confirmation
            elif curr_low < S3_aligned[i] and bear_regime and vol_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: exit when price crosses below Camarilla R2
            if curr_close < R2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price crosses above Camarilla S2
            if curr_close > S2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S3_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0