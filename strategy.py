#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_1d_parabolic_sar_trend_v1
# Parabolic SAR on 6h with 1d ADX trend filter (ADX > 25).
# Captures strong trends while avoiding whipsaws in ranging markets.
# Works in both bull and bear by following trend direction.
# Target: 50-150 total trades over 4 years (12-37/year).
name = "6h_1d_parabolic_sar_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ADX
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.inf], np.maximum.reduce([tr1, tr2, tr3])])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    def wilder_smooth(data, period):
        smoothed = np.zeros_like(data)
        smoothed[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + data[i]
        return smoothed
    
    period = 14
    tr_smoothed = wilder_smooth(tr, period)
    dm_plus_smoothed = wilder_smooth(dm_plus, period)
    dm_minus_smoothed = wilder_smooth(dm_minus, period)
    
    # DI+ and DI-
    di_plus = np.where(tr_smoothed != 0, 100 * dm_plus_smoothed / tr_smoothed, 0)
    di_minus = np.where(tr_smoothed != 0, 100 * dm_minus_smoothed / tr_smoothed, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilder_smooth(dx, period)
    
    # Align ADX to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Parabolic SAR on 6h
    def calculate_psar(high, low, af_start=0.02, af_increment=0.02, af_max=0.2):
        psar = np.zeros_like(high)
        psar[0] = low[0]
        up_trend = True
        af = af_start
        ep = high[0] if up_trend else low[0]
        
        for i in range(1, len(high)):
            if up_trend:
                psar[i] = psar[i-1] + af * (ep - psar[i-1])
                # Prevent SAR from penetrating previous lows
                psar[i] = min(psar[i], low[i-1], low[i-2] if i >= 2 else low[i-1])
                if low[i] < psar[i]:  # Trend reversal
                    up_trend = False
                    psar[i] = ep
                    af = af_start
                    ep = low[i]
                else:
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + af_increment, af_max)
            else:
                psar[i] = psar[i-1] + af * (ep - psar[i-1])
                # Prevent SAR from penetrating previous highs
                psar[i] = max(psar[i], high[i-1], high[i-2] if i >= 2 else high[i-1])
                if high[i] > psar[i]:  # Trend reversal
                    up_trend = True
                    psar[i] = ep
                    af = af_start
                    ep = high[i]
                else:
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + af_increment, af_max)
        return psar
    
    psar = calculate_psar(high, low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if ADX not ready
        if np.isnan(adx_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: only trade when ADX > 25 (strong trend)
        strong_trend = adx_aligned[i] > 25
        
        # PSAR signals: price above SAR = bullish, below SAR = bearish
        bullish_signal = strong_trend and close[i] > psar[i]
        bearish_signal = strong_trend and close[i] < psar[i]
        
        # Exit when trend weakens or PSAR flips
        exit_long = not strong_trend or close[i] < psar[i]
        exit_short = not strong_trend or close[i] > psar[i]
        
        if bullish_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals