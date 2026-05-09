#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Donchian20_Volume_Spike_RsiTrend"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = pd.DatetimeIndex(prices['open_time'])
    hours = open_time.hour
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate EMA20 on 4h close for trend filter
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Get 1d data for Donchian channel
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) on 1d high/low
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    high_max = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high = align_htf_to_ltf(prices, df_1d, high_max)
    donchian_low = align_htf_to_ltf(prices, df_1d, low_min)
    
    # Volume spike filter: current volume > 2.0 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # RSI(14) for momentum filter
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Need enough data for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema20_4h_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(rsi_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema20 = ema20_4h_aligned[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        vol_spike = volume_spike[i]
        rsi_val = rsi_values[i]
        hour = hours[i]
        
        # Session filter: 08-20 UTC
        in_session = (8 <= hour <= 20)
        
        if position == 0:
            # Enter long: Close breaks above Donchian high + 4h uptrend + volume spike + RSI > 50 + session
            if in_session and close[i] > upper and close[i] > ema20 and vol_spike and rsi_val > 50:
                signals[i] = 0.20
                position = 1
            # Enter short: Close breaks below Donchian low + 4h downtrend + volume spike + RSI < 50 + session
            elif in_session and close[i] < lower and close[i] < ema20 and vol_spike and rsi_val < 50:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: Close falls below Donchian low or 4h trend turns down
            if close[i] < lower or close[i] < ema20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: Close rises above Donchian high or 4h trend turns up
            if close[i] > upper or close[i] > ema20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals