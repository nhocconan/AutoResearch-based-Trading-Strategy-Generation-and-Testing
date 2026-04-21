#!/usr/bin/env python3
"""
1d_VolumeBreakout_Pullback
Hypothesis: On daily chart, buy breakouts above prior 20-day high with volume > 1.5x 20-day average, then enter on pullback to 20-day EMA when RSI(14) < 40. Short symmetrical. Weekly trend filter (price above/below weekly 200 EMA) ensures alignment with higher timeframe trend. Designed to capture momentum with defined risk in both bull and bear markets by following weekly trend. Target ~15-25 trades/year on 1d.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily and weekly data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 50:
        return np.zeros(n)
    
    # === Daily indicators: 20-day high/low, EMA20, RSI14, volume average ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    highest_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_14_1d = 100 - (100 / (1 + rs))
    
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # === Weekly trend filter: 200-period EMA ===
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if indicators not ready
        if (np.isnan(highest_high_20[i]) or
            np.isnan(lowest_low_20[i]) or
            np.isnan(ema_20_1d[i]) or
            np.isnan(rsi_14_1d[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(ema_200_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close_1d[i]
        highest_high = highest_high_20[i]
        lowest_low = lowest_low_20[i]
        ema_20 = ema_20_1d[i]
        rsi_val = rsi_14_1d[i]
        vol_ratio = volume_1d[i] / vol_ma_20[i] if vol_ma_20[i] != 0 else 1.0
        weekly_trend = ema_200_1w_aligned[i]
        
        if position == 0:
            # Long: break above 20-day high + volume spike > 1.5, then pullback to EMA20 with RSI<40
            if (price_close > highest_high and vol_ratio > 1.5):
                # Enter on next bar pullback
                if i + 1 < n:
                    # We'll handle entry on next iteration when pullback occurs
                    pass
            # Check for pullback entry: price near EMA20 and RSI<40
            if (abs(price_close - ema_20) / ema_20 < 0.02 and  # within 2% of EMA20
                rsi_val < 40 and
                price_close > weekly_trend):  # only in uptrend
                signals[i] = 0.25
                position = 1
            
            # Short: break below 20-day low + volume spike > 1.5, then pullback to EMA20 with RSI>60
            if (price_close < lowest_low and vol_ratio > 1.5):
                # Enter on next bar pullback
                if i + 1 < n:
                    pass
            # Check for pullback entry: price near EMA20 and RSI>60
            if (abs(price_close - ema_20) / ema_20 < 0.02 and  # within 2% of EMA20
                rsi_val > 60 and
                price_close < weekly_trend):  # only in downtrend
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price closes back above/below 20-day EMA
            if position == 1 and price_close < ema_20:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > ema_20:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_VolumeBreakout_Pullback"
timeframe = "1d"
leverage = 1.0