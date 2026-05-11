#!/usr/bin/env python3
name = "1h_Camarilla_R1S3_Breakout_4hTrend_1dVolatilityFilter"
timeframe = "1h"
leverage = 1.0

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
    
    # Get daily data for volatility filter (ATR ratio)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily ATR(14)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily ATR ratio: ATR(7)/ATR(14) to detect volatility expansion/contraction
    atr_7_1d = pd.Series(tr).rolling(window=7, min_periods=7).mean().values
    atr_ratio = np.where(atr_1d != 0, atr_7_1d / atr_1d, 1.0)
    atr_ratio = np.nan_to_num(atr_ratio, nan=1.0)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Get 4h data for Camarilla levels and trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Previous 4h bar (to avoid look-ahead)
    prev_high = np.roll(high_4h, 1)
    prev_low = np.roll(low_4h, 1)
    prev_close = np.roll(close_4h, 1)
    prev_high[0] = high_4h[0]
    prev_low[0] = low_4h[0]
    prev_close[0] = close_4h[0]
    
    # Camarilla levels for previous 4h bar
    R3 = prev_close + 1.1 * (prev_high - prev_low) / 6
    R1 = prev_close + 1.1 * (prev_high - prev_low) / 12
    S1 = prev_close - 1.1 * (prev_high - prev_low) / 12
    S3 = prev_close - 1.1 * (prev_high - prev_low) / 6
    
    # Align Camarilla levels to 1h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_4h, R3)
    R1_aligned = align_htf_to_ltf(prices, df_4h, R1)
    S1_aligned = align_htf_to_ltf(prices, df_4h, S1)
    S3_aligned = align_htf_to_ltf(prices, df_4h, S3)
    
    # 4h EMA34 for trend filter
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # 1h volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma != 0, volume / vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when volatility is contracting (ATR ratio < 0.8)
        volatility_contracting = atr_ratio_aligned[i] < 0.8
        
        if position == 0:
            # Long: Price breaks above R1 with volume, volatility contracting, and above 4h EMA34
            if (close[i] > R1_aligned[i] and 
                vol_ratio[i] > 1.5 and 
                volatility_contracting and 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below S1 with volume, volatility contracting, and below 4h EMA34
            elif (close[i] < S1_aligned[i] and 
                  vol_ratio[i] > 1.5 and 
                  volatility_contracting and 
                  close[i] < ema_34_aligned[i]):
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions: price reaches opposite Camarilla level or trend fails
            if position == 1:
                # Exit long: price reaches S3 or closes below 4h EMA34
                if (close[i] <= S3_aligned[i]) or (close[i] < ema_34_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Exit short: price reaches R3 or closes above 4h EMA34
                if (close[i] >= R3_aligned[i]) or (close[i] > ema_34_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals