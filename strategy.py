#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and 1w EMA(50) trend filter
# - Long when price breaks above Camarilla H3 level AND 1d volume > 1.5x 20-period average AND 1w close > 1w EMA(50)
# - Short when price breaks below Camarilla L3 level AND 1d volume > 1.5x 20-period average AND 1w close < 1w EMA(50)
# - Exit when price crosses Camarilla Pivot point (midpoint)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Camarilla levels provide intraday structure; volume confirms breakout validity
# - Weekly EMA filter ensures we trade with the higher timeframe trend
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)

name = "12h_1d_1w_camarilla_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Pre-compute 12h OHLC
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 12h Camarilla levels (based on previous day's OHLC)
    def calculate_camarilla(h_prev, l_prev, c_prev):
        range_ = h_prev - l_prev
        pivot = (h_prev + l_prev + c_prev) / 3.0
        h3 = pivot + (range_ * 1.1 / 4)
        l3 = pivot - (range_ * 1.1 / 4)
        return pivot, h3, l3
    
    camarilla_pivot = np.full_like(close, np.nan, dtype=float)
    camarilla_h3 = np.full_like(close, np.nan, dtype=float)
    camarilla_l3 = np.full_like(close, np.nan, dtype=float)
    
    # Use previous day's OHLC for today's Camarilla levels
    for i in range(1, len(prices)):
        h_prev = high[i-1]
        l_prev = low[i-1]
        c_prev = close[i-1]
        pivot, h3, l3 = calculate_camarilla(h_prev, l_prev, c_prev)
        camarilla_pivot[i] = pivot
        camarilla_h3[i] = h3
        camarilla_l3[i] = l3
    
    # Pre-compute 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    def rolling_mean(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.mean(arr[i - window + 1:i + 1])
        return result
    
    vol_ma_1d = rolling_mean(volume_1d, 20)
    
    # Pre-compute 1w EMA(50)
    close_1w = df_1w['close'].values
    ema_50_1w = np.full_like(close_1w, np.nan, dtype=float)
    if len(close_1w) >= 50:
        ema_50_1w[49] = np.mean(close_1w[:50])  # SMA seed
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = (close_1w[i] * 2 + ema_50_1w[i-1] * 48) / 50  # EMA(50)
    
    # Align HTF indicators to 12h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or np.isnan(camarilla_pivot[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Volume spike condition
            vol_ma_12h = rolling_mean(volume, 20)
            vol_spike = not np.isnan(vol_ma_12h[i]) and volume[i] > 1.5 * vol_ma_12h[i]
            
            # Long conditions: Camarilla H3 breakout AND volume spike AND 1w uptrend
            if (close[i] > camarilla_h3[i] and vol_spike and 
                close_1w[-1] > ema_50_1w_aligned[i] if len(close_1w) > 0 else False):
                position = 1
                signals[i] = 0.25
            # Short conditions: Camarilla L3 breakdown AND volume spike AND 1w downtrend
            elif (close[i] < camarilla_l3[i] and vol_spike and 
                  close_1w[-1] < ema_50_1w_aligned[i] if len(close_1w) > 0 else False):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price crosses Camarilla Pivot point
            exit_long = (position == 1 and close[i] < camarilla_pivot[i])
            exit_short = (position == -1 and close[i] > camarilla_pivot[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals

def rolling_mean(arr, window):
    result = np.full_like(arr, np.nan, dtype=float)
    for i in range(window - 1, len(arr)):
        result[i] = np.mean(arr[i - window + 1:i + 1])
    return result