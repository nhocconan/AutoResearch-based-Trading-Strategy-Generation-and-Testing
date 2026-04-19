#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot reversal with 12h ADX filter and volume confirmation
# Uses 12h ADX to filter strong trends (ADX > 25) and only trade reversals in ranging markets (ADX < 25)
# Camarilla levels from 1d: long at S1/S2, short at R1/R2 with rejection candlestick
# Volume confirmation: current volume > 1.5x 20-period average
# Target: 15-25 trades/year per symbol with high-probability reversals in ranging markets
name = "6h_Camarilla_12hADX_Reversal_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h ADX for trend strength filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    # Calculate ADX on 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with original index
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    def smooth_wilder(data, period):
        smoothed = np.full_like(data, np.nan, dtype=float)
        if len(data) >= period:
            smoothed[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                smoothed[i] = smoothed[i-1] - (smoothed[i-1]/period) + data[i]
        return smoothed
    
    period = 14
    atr = smooth_wilder(tr, period)
    dm_plus_smooth = smooth_wilder(dm_plus, period)
    dm_minus_smooth = smooth_wilder(dm_minus, period)
    
    # DI and DX
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = smooth_wilder(dx, period)
    adx_12h = align_htf_to_ltf(prices, df_12h, adx)
    
    # 1d Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels
    R4 = prev_close + (prev_high - prev_low) * 1.1 / 2
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    R2 = prev_close + (prev_high - prev_low) * 1.1 / 6
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    S2 = prev_close - (prev_high - prev_low) * 1.1 / 6
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    S4 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    R1_1d = align_htf_to_ltf(prices, df_1d, R1)
    R2_1d = align_htf_to_ltf(prices, df_1d, R2)
    S1_1d = align_htf_to_ltf(prices, df_1d, S1)
    S2_1d = align_htf_to_ltf(prices, df_1d, S2)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_12h[i]) or np.isnan(R1_1d[i]) or np.isnan(R2_1d[i]) or 
            np.isnan(S1_1d[i]) or np.isnan(S2_1d[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in ranging markets (ADX < 25)
        if adx_12h[i] >= 25:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price at S1 or S2 with bullish rejection
            if ((close[i] <= S1_1d[i] * 1.001 and close[i] >= S1_1d[i] * 0.999) or
                (close[i] <= S2_1d[i] * 1.001 and close[i] >= S2_1d[i] * 0.999)):
                # Bullish rejection: close > open and close > (high+low)/2
                if close[i] > prices['open'].iloc[i] and close[i] > (high[i] + low[i]) / 2:
                    if volume_confirm[i]:
                        signals[i] = 0.25
                        position = 1
            # Short: price at R1 or R2 with bearish rejection
            elif ((close[i] >= R1_1d[i] * 0.999 and close[i] <= R1_1d[i] * 1.001) or
                  (close[i] >= R2_1d[i] * 0.999 and close[i] <= R2_1d[i] * 1.001)):
                # Bearish rejection: close < open and close < (high+low)/2
                if close[i] < prices['open'].iloc[i] and close[i] < (high[i] + low[i]) / 2:
                    if volume_confirm[i]:
                        signals[i] = -0.25
                        position = -1
                        
        elif position == 1:
            # Long: exit if price reaches R1 or ADX strengthens
            if (close[i] >= R1_1d[i] * 0.999) or (adx_12h[i] >= 25):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price reaches S1 or ADX strengthens
            if (close[i] <= S1_1d[i] * 1.001) or (adx_12h[i] >= 25):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals