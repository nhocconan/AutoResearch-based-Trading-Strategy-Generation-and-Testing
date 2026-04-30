#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Camarilla pivot levels from 4h provide precise intraday support/resistance based on prior 4h range
# Breakout above R3 or below S3 with volume confirmation indicates strong momentum
# 1d EMA34 ensures alignment with daily trend to avoid counter-trend trades
# Volume spike (2.0x 24-period average) confirms institutional participation
# Session filter (08-20 UTC) reduces noise trades
# Discrete sizing 0.20 minimizes fee churn. Target: 60-150 total trades over 4 years (15-37/year).

name = "1h_Camarilla_R3S3_1dEMA34_VolumeSpike_v1"
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
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h Camarilla pivot levels (R3, S3)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    # Camarilla levels based on prior 4h's OHLC
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla R3 and S3
    camarilla_r3 = close_4h + ((high_4h - low_4h) * 1.25 / 2)
    camarilla_s3 = close_4h - ((high_4h - low_4h) * 1.25 / 2)
    
    # Align to 1h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # Volume confirmation: volume > 2.0x 24-period average (24*1h = 24h)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(34, 2)  # warmup for 1d EMA34 and 4h data
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
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
        curr_ema_34 = ema_34_1d_aligned[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish entry: break above R3 with close > R3 AND price > 1d EMA34 (uptrend)
                if curr_close > curr_r3 and curr_close > curr_ema_34:
                    signals[i] = 0.20
                    position = 1
                    entry_price = curr_close
                # Bearish entry: break below S3 with close < S3 AND price < 1d EMA34 (downtrend)
                elif curr_close < curr_s3 and curr_close < curr_ema_34:
                    signals[i] = -0.20
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price drops below R3 (breakout fails) OR drops below 1d EMA34 (trend change)
            if curr_close < curr_r3 or curr_close < curr_ema_34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit when price rises above S3 (breakdown fails) OR rises above 1d EMA34 (trend change)
            if curr_close > curr_s3 or curr_close > curr_ema_34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals