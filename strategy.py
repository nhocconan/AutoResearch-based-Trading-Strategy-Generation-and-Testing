#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume confirmation
# Camarilla pivot levels provide high-probability intraday reversal/breakout zones
# 1w EMA > 34 ensures we only trade in the direction of the weekly trend
# Volume spike confirms institutional participation behind the breakout
# Designed for very low frequency (30-100 trades over 4 years) to minimize fee drag
# Works in bull/bear via trend filter + breakout logic

name = "1d_Camarilla_R3_S3_Breakout_1wEMA34_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivot calculation (use same timeframe as primary)
    df_1d = prices.copy()  # primary timeframe is 1d
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1w HTF data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(30, 20)  # Need 1w EMA34 and volume MA20
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla levels for today using yesterday's OHLC
        if i == 0:
            signals[i] = 0.0
            continue
            
        # Yesterday's OHLC for Camarilla calculation
        y_high = high[i-1]
        y_low = low[i-1]
        y_close = close[i-1]
        
        # Camarilla pivot levels
        pivot = (y_high + y_low + y_close) / 3
        range_hl = y_high - y_low
        
        # Resistance and Support levels
        r3 = pivot + (range_hl * 1.1 / 4)
        s3 = pivot - (range_hl * 1.1 / 4)
        
        # Trend filter: price > 1w EMA34 for long, price < 1w EMA34 for short
        uptrend = close[i] > ema_34_1w_aligned[i]
        downtrend = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R3 with volume spike in uptrend
            if close[i] > r3 and volume_spike[i] and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume spike in downtrend
            elif close[i] < s3 and volume_spike[i] and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on close below pivot or trend reversal
            if close[i] < pivot or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on close above pivot or trend reversal
            if close[i] > pivot or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals