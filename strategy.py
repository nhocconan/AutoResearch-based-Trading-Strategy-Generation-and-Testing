#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R4/S4 breakout with 4h EMA50 trend filter and volume spike confirmation
# Camarilla R4/S4 levels act as strong intraday support/resistance where breakouts often continue
# 4h EMA50 filter ensures we only trade in the direction of the higher timeframe trend
# Volume spike (2.0x 20-period average) confirms institutional participation and reduces false breakouts
# Works in bull markets via breakouts above R4 and bear markets via breakdowns below S4
# Discrete sizing 0.20 minimizes fee churn. Target: 60-150 total trades over 4 years (15-37/year).
# Uses 1h timeframe with 4h/1d HTF for direction, 1h only for entry timing precision.

name = "1h_Camarilla_R4_S4_Breakout_4hEMA50_VolumeSpike_v1"
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
    
    # Load 4h data ONCE before loop (MTF Rule #1)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 4h Camarilla pivot levels (R4, S4)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    daily_range = high_4h - low_4h
    camarilla_r4 = close_4h + daily_range * 1.1 / 2
    camarilla_s4 = close_4h - daily_range * 1.1 / 2
    
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s4)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 50, 20)  # warmup for EMA50, Camarilla, and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_50_4h = ema_50_4h_aligned[i]
        curr_camarilla_r4 = camarilla_r4_aligned[i]
        curr_camarilla_s4 = camarilla_s4_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish entry: break above Camarilla R4 AND above 4h EMA50 (uptrend)
                if curr_high > curr_camarilla_r4 and curr_close > curr_ema_50_4h:
                    signals[i] = 0.20
                    position = 1
                    entry_price = curr_close
                # Bearish entry: break below Camarilla S4 AND below 4h EMA50 (downtrend)
                elif curr_low < curr_camarilla_s4 and curr_close < curr_ema_50_4h:
                    signals[i] = -0.20
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price drops below Camarilla S4 (breakout fails)
            if curr_close < curr_camarilla_s4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit when price rises above Camarilla R4 (breakdown fails)
            if curr_close > curr_camarilla_r4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals