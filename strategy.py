#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and 1h momentum filter
# - Long when price breaks above 4h Donchian H20 AND 1d volume > 1.5x 20-period average AND 1h RSI > 50
# - Short when price breaks below 4h Donchian L20 AND 1d volume > 1.5x 20-period average AND 1h RSI < 50
# - Exit when price returns to 4h Donchian midpoint or reverses to opposite band
# - Uses discrete position sizing 0.25 to limit fee churn
# - Donchian channels provide objective breakout levels based on price structure
# - Volume confirmation reduces false breakouts in low-participation moves
# - 1h RSI filter ensures alignment with short-term momentum to avoid counter-trend entries
# - Target: 20-40 trades/year on 4h timeframe (80-160 total over 4 years)

name = "4h_1d_1h_donchian_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 4h Donchian channels (20-period)
    lookback = 20
    donchian_h = np.full_like(high, np.nan, dtype=float)
    donchian_l = np.full_like(low, np.nan, dtype=float)
    for i in range(lookback - 1, len(high)):
        donchian_h[i] = np.max(high[i - lookback + 1:i + 1])
        donchian_l[i] = np.min(low[i - lookback + 1:i + 1])
    donchian_mid = (donchian_h + donchian_l) / 2.0
    
    # Pre-compute 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    vol_ma_1d = np.full_like(volume_1d, np.nan, dtype=float)
    for i in range(19, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i - 19:i + 1])
    
    # Pre-compute 1h RSI(14) - needs HTF data
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 30:
        return np.zeros(n)
    close_1h = df_1h['close'].values
    
    # RSI calculation
    delta = np.diff(close_1h, prepend=close_1h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    def wilder_smoothing(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        result[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    avg_gain = wilder_smoothing(gain, 14)
    avg_loss = wilder_smoothing(loss, 14)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1h = 100 - (100 / (1 + rs))
    
    # Align HTF indicators to 4h timeframe
    donchian_h_aligned = align_htf_to_ltf(prices, prices, donchian_h)  # 4h to 4h is identity
    donchian_l_aligned = align_htf_to_ltf(prices, prices, donchian_l)
    donchian_mid_aligned = align_htf_to_ltf(prices, prices, donchian_mid)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_h_aligned[i]) or np.isnan(donchian_l_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(rsi_1h_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Volume spike condition (1.5x average)
            vol_spike = volume[i] > 1.5 * vol_ma_1d_aligned[i]
            
            # Long conditions: price > Donchian H20 AND volume spike AND 1h RSI > 50
            if (close[i] > donchian_h_aligned[i] and vol_spike and rsi_1h_aligned[i] > 50):
                position = 1
                signals[i] = 0.25
            # Short conditions: price < Donchian L20 AND volume spike AND 1h RSI < 50
            elif (close[i] < donchian_l_aligned[i] and vol_spike and rsi_1h_aligned[i] < 50):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price returns to midpoint or reverses to opposite band
            exit_long = (position == 1 and (close[i] < donchian_mid_aligned[i] or close[i] < donchian_l_aligned[i]))
            exit_short = (position == -1 and (close[i] > donchian_mid_aligned[i] or close[i] > donchian_h_aligned[i]))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals