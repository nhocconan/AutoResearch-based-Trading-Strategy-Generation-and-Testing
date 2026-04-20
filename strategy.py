#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4d_Camarilla_R3S3_Breakout_VolumeTrend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 2 or len(df_1d) < 2:
        return np.zeros(n)
    
    # === 1d: Calculate Camarilla pivot points ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # Range = H - L
    range_1d = high_1d - low_1d
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    r3_1d = close_1d + range_1d * 1.1 / 2.0
    s3_1d = close_1d - range_1d * 1.1 / 2.0
    
    # Align Camarilla levels
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # === 4h: EMA34 for trend filter ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    close_4h_series = pd.Series(close_4h)
    ema34_4h = close_4h_series.ewm(span=34, min_periods=34, adjust=False).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # === 1h: Price, volume, ATR ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR(14)
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if outside session
        if not session_mask[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get aligned values
        r3 = r3_1d_aligned[i]
        s3 = s3_1d_aligned[i]
        ema34 = ema34_4h_aligned[i]
        current_atr = atr[i]
        current_close = close[i]
        current_volume = volume[i]
        
        # Skip if any value is NaN
        if (np.isnan(r3) or np.isnan(s3) or np.isnan(ema34) or np.isnan(current_atr)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.5x 20-period average
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
            vol_condition = current_volume > 1.5 * vol_ma
        else:
            vol_condition = False
        
        if position == 0:
            # Long: break above R3 with volume AND above 4h EMA34 (uptrend)
            if current_close > r3 and vol_condition and current_close > ema34:
                signals[i] = 0.20
                position = 1
                entry_price = current_close
            
            # Short: break below S3 with volume AND below 4h EMA34 (downtrend)
            elif current_close < s3 and vol_condition and current_close < ema34:
                signals[i] = -0.20
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Long exit: price fails to hold above R3 OR stop loss
            if current_close <= r3 or current_close < entry_price - 2.0 * current_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price fails to hold below S3 OR stop loss
            if current_close >= s3 or current_close > entry_price + 2.0 * current_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals