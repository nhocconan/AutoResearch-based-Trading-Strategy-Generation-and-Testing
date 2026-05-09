#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout (20) with 12h EMA50 trend filter and volume confirmation.
# Long when price breaks above upper Donchian + price > 12h EMA50 + volume > 1.5x average.
# Short when price breaks below lower Donchian + price < 12h EMA50 + volume > 1.5x average.
# Exit when price crosses opposite Donchian band or volume drops below average.
# Designed to capture strong trend moves with volume confirmation, avoiding false breakouts.
# Target: 20-40 trades/year per symbol.
name = "4h_Donchian20_12hEMA50_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

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
    
    # Calculate 50-period EMA on 12h close
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channels (20-period)
    donchian_len = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(donchian_len - 1, n):
        upper[i] = np.max(high[i-donchian_len+1:i+1])
        lower[i] = np.min(low[i-donchian_len+1:i+1])
    
    # Calculate volume average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(49, donchian_len - 1, 19)  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if np.isnan(ema_50_12h_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema = ema_50_12h_aligned[i]
        up = upper[i]
        low_ch = lower[i]
        vol = volume[i]
        vol_avg = vol_ma[i]
        
        if position == 0:
            # Enter long: price breaks above upper Donchian + price > 12h EMA50 + volume > 1.5x average
            if price > up and price > ema and vol > 1.5 * vol_avg:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Donchian + price < 12h EMA50 + volume > 1.5x average
            elif price < low_ch and price < ema and vol > 1.5 * vol_avg:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below lower Donchian OR volume drops below average
            if price < low_ch or vol < vol_avg:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above upper Donchian OR volume drops below average
            if price > up or vol < vol_avg:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals