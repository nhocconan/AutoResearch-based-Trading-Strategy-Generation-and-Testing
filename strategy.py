#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_12hEMA50_Trend_VolumeSpike_ATRStop
Hypothesis: Camarilla R3/S3 breakout with 12h EMA50 trend filter, volume spike confirmation, and ATR trailing stop.
Long when price breaks above R3, 12h EMA50 uptrend, and volume > 2x average.
Short when price breaks below S3, 12h EMA50 downtrend, and volume > 2x average.
Uses ATR-based trailing stop (2.5x ATR from extreme) to manage risk.
Designed for low trade frequency (19-50/year) to avoid fee drag while capturing momentum in both bull and bear markets.
Uses discrete position sizing (0.30) to minimize fee churn.
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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R3, R2, R1, PP, S1, S2, S3
    # R3 = close + 1.1*(high-low)*1.1/4
    # S3 = close - 1.1*(high-low)*1.1/4
    camarilla_range = (high_1d - low_1d) * 1.1
    camarilla_r3 = close_1d + camarilla_range * 1.1 / 4
    camarilla_s3 = close_1d - camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (1-day delay for completed bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # ATR for volatility and stop loss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: 2x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    
    # Warmup: max of 12h EMA (50), 1d data (2), ATR (14), volume MA (20)
    start_idx = max(50, 2, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        ema_50_12h_val = ema_50_12h_aligned[i]
        camarilla_r3_val = camarilla_r3_aligned[i]
        camarilla_s3_val = camarilla_s3_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price breaks above R3, uptrend, volume spike
            long_signal = (close_val > camarilla_r3_val) and (ema_50_12h_val > ema_50_12h_aligned[i-1]) and (volume_val > 2.0 * vol_ma_val)
            # Short: price breaks below S3, downtrend, volume spike
            short_signal = (close_val < camarilla_s3_val) and (ema_50_12h_val < ema_50_12h_aligned[i-1]) and (volume_val > 2.0 * vol_ma_val)
            
            if long_signal:
                signals[i] = 0.30
                position = 1
                entry_price = close_val
                long_stop = entry_price - 2.5 * atr_val
            elif short_signal:
                signals[i] = -0.30
                position = -1
                entry_price = close_val
                short_stop = entry_price + 2.5 * atr_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.30
            # Update trailing stop: move stop up as price makes new highs
            long_stop = max(long_stop, high_val - 2.5 * atr_val)
            # Exit: trailing stop hit or trend reversal
            if (low_val < long_stop) or (ema_50_12h_val < ema_50_12h_aligned[i-1]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.30
            # Update trailing stop: move stop down as price makes new lows
            short_stop = min(short_stop, low_val + 2.5 * atr_val)
            # Exit: trailing stop hit or trend reversal
            if (high_val > short_stop) or (ema_50_12h_val > ema_50_12h_aligned[i-1]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_12hEMA50_Trend_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0