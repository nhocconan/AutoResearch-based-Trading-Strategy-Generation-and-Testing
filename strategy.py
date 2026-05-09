#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Channels strategy using 1-day ATR-based channels with 1-week trend filter and volume confirmation.
# Enters long when price breaks above upper channel with weekly uptrend and volume spike, short when price breaks below lower channel with weekly downtrend and volume spike.
# Uses daily timeframe for channel calculation to avoid look-ahead and weekly trend for filter to avoid whipsaw.
# Designed to work in both bull and bear markets by aligning with weekly trend. Target: 15-30 trades/year to minimize fee drag.

name = "1d_ATR_Channel_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR channel calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate ATR(14) on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR-based channels (similar to Keltner)
    ma20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    upper_channel = ma20 + (2.0 * atr14)
    lower_channel = ma20 - (2.0 * atr14)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate EMA20 on 1w close for trend filter
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Align daily indicators to 1d timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_channel)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_channel)
    ma20_aligned = align_htf_to_ltf(prices, df_1d, ma20)
    
    # Volume spike filter: current volume > 2.0 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20, 14)  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema20_1w_aligned[i]) or 
            np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or
            np.isnan(ma20_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema20_1w_val = ema20_1w_aligned[i]
        upper = upper_aligned[i]
        lower = lower_aligned[i]
        ma20_val = ma20_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Close breaks above upper channel + weekly uptrend + volume spike
            if close[i] > upper and close[i] > ema20_1w_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: Close breaks below lower channel + weekly downtrend + volume spike
            elif close[i] < lower and close[i] < ema20_1w_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Close falls below middle line or weekly trend turns down
            if close[i] < ma20_val or close[i] < ema20_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close rises above middle line or weekly trend turns up
            if close[i] > ma20_val or close[i] > ema20_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals