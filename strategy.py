#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume spike confirmation
# Camarilla pivot levels provide precise intraday support/resistance based on prior day's range
# Breakout above R1 or below S1 with volume confirmation indicates strong momentum
# 4h EMA50 ensures alignment with intermediate trend to avoid counter-trend trades
# Volume spike (2.0x 24-period average) confirms institutional participation
# Session filter (08-20 UTC) reduces noise trades
# Discrete sizing 0.20 minimizes fee churn. Target: 60-150 total trades over 4 years (15-37/year).

name = "1h_Camarilla_R1S1_4hEMA50_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d Camarilla pivot levels (R1, S1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    # Camarilla levels based on prior day's OHLC
    # R1 = C + ((H-L) * 1.083/2), S1 = C - ((H-L) * 1.083/2)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla R1 and S1
    camarilla_r1 = close_1d + ((high_1d - low_1d) * 1.083 / 2)
    camarilla_s1 = close_1d - ((high_1d - low_1d) * 1.083 / 2)
    
    # Align to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: volume > 2.0x 24-period average (24 hours = 1 day)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 2)  # warmup for 4h EMA50 and 1d data
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_50 = ema_50_4h_aligned[i]
        curr_r1 = camarilla_r1_aligned[i]
        curr_s1 = camarilla_s1_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish entry: break above R1 with close > R1 AND price > 4h EMA50 (uptrend)
                if curr_close > curr_r1 and curr_close > curr_ema_50:
                    signals[i] = 0.20
                    position = 1
                    entry_price = curr_close
                # Bearish entry: break below S1 with close < S1 AND price < 4h EMA50 (downtrend)
                elif curr_close < curr_s1 and curr_close < curr_ema_50:
                    signals[i] = -0.20
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price drops below R1 (breakout fails) OR drops below 4h EMA50 (trend change)
            if curr_close < curr_r1 or curr_close < curr_ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit when price rises above S1 (breakdown fails) OR rises above 4h EMA50 (trend change)
            if curr_close > curr_s1 or curr_close > curr_ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals