#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume_v2"
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
    
    # Get 1d data for trend filter and pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Camarilla R3, S3 levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    camarilla_range = (high_1d - low_1d) * 1.1 / 6
    r3_level = close_1d_vals + camarilla_range * 4
    s3_level = close_1d_vals - camarilla_range * 4
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_level)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_level)
    
    # Volume spike filter: current volume > 1.8 * 50-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    # RSI filter on 4h: avoid overbought/oversold extremes
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    rsi_not_extreme = (rsi_values > 20) & (rsi_values < 80)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 50)  # Need enough data for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(rsi_not_extreme[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50 = ema50_1d_aligned[i]
        r3 = r3_aligned[i]
        s3 = s3_aligned[i]
        vol_spike = volume_spike[i]
        rsi_ok = rsi_not_extreme[i]
        
        if position == 0:
            # Enter long: Close breaks above R3 + 1d uptrend + volume spike + RSI not extreme
            if close[i] > r3 and close[i] > ema50 and vol_spike and rsi_ok:
                signals[i] = 0.25
                position = 1
            # Enter short: Close breaks below S3 + 1d downtrend + volume spike + RSI not extreme
            elif close[i] < s3 and close[i] < ema50 and vol_spike and rsi_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Close falls below S3 or 1d trend turns down
            if close[i] < s3 or close[i] < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close rises above R3 or 1d trend turns up
            if close[i] > r3 or close[i] > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals