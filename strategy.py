#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA34 trend filter and volume confirmation.
Long when price breaks above Camarilla R3 AND close > 4h EMA34 AND volume > 2.0x 24-period average.
Short when price breaks below Camarilla S3 AND close < 4h EMA34 AND volume > 2.0x 24-period average.
Exit when price retraces to Camarilla pivot point (PP).
Uses discrete position sizing (0.20) targeting ~20-40 trades/year on 1h timeframe.
Camarilla R3/S3 represent stronger intraday support/resistance levels; breakouts with volume and trend alignment capture momentum moves while reducing false signals.
Session filter (08-20 UTC) avoids low-liquidity periods. Works in bull (trend-following breakouts) and bear (mean-reversion via volatility expansion at extremes).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h EMA34 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Calculate Camarilla pivot levels from 1d (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: PP = (H+L+C)/3, R3 = PP + (H-L)*1.1/4, S3 = PP - (H-L)*1.1/4
    pivot = (high_1d + low_1d + close_1d) / 3.0
    camarilla_r3 = pivot + (high_1d - low_1d) * 1.1 / 4.0
    camarilla_s3 = pivot - (high_1d - low_1d) * 1.1 / 4.0
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Volume average (24-period = 1 day at 1h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Session filter: 08-20 UTC (avoid low liquidity periods)
    hours = prices.index.hour  # open_time is already datetime64[ms]
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 24)  # EMA34 needs 34, vol MA needs 24
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema34_4h_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or np.isnan(pivot_aligned[i]) or
            np.isnan(vol_ma[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        ema34_val = ema34_4h_aligned[i]
        r3_val = camarilla_r3_aligned[i]
        s3_val = camarilla_s3_aligned[i]
        pivot_val = pivot_aligned[i]
        
        if position == 0:
            # Long: Break above Camarilla R3 AND uptrend (price > EMA34) AND volume spike (2.0x avg)
            if close[i] > r3_val and close[i] > ema34_val and volume[i] > 2.0 * vol_ma_val:
                signals[i] = 0.20
                position = 1
            # Short: Break below Camarilla S3 AND downtrend (price < EMA34) AND volume spike (2.0x avg)
            elif close[i] < s3_val and close[i] < ema34_val and volume[i] > 2.0 * vol_ma_val:
                signals[i] = -0.20
                position = -1
        else:
            # Exit when price retraces to Camarilla pivot point
            exit_signal = False
            if position == 1 and close[i] <= pivot_val:
                exit_signal = True
            elif position == -1 and close[i] >= pivot_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R3S3_Breakout_4hEMA34_Trend_VolumeConfirmation_PivotExit_SessionFilter"
timeframe = "1h"
leverage = 1.0