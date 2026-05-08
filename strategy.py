#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeFilter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 1w ATR for volatility filter
    tr_w = np.maximum(high_1w - low_1w, 
                      np.maximum(np.abs(high_1w - np.roll(close_1w, 1)), 
                                 np.abs(low_1w - np.roll(close_1w, 1))))
    tr_w[0] = high_1w[0] - low_1w[0]
    atr14_1w = pd.Series(tr_w).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr14_1w)
    
    # 1d data for Camarilla pivot (previous day)
    prev_high_1d = np.roll(high, 1)
    prev_low_1d = np.roll(low, 1)
    prev_close_1d = np.roll(close, 1)
    prev_high_1d[0] = high[0]  # first bar uses current
    prev_low_1d[0] = low[0]
    prev_close_1d[0] = close[0]
    
    pivot = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    range_1d = prev_high_1d - prev_low_1d
    r1 = pivot + (range_1d * 1.1 / 12)
    s1 = pivot - (range_1d * 1.1 / 12)
    
    # Volume filter: 1d volume > 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # warmup for EMA50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(atr14_1w_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1, price above 1w EMA50, volume above average
            long_cond = (close[i] > r1[i] and 
                        close[i] > ema50_1w_aligned[i] and
                        volume[i] > vol_ma20[i])
            
            # Short: Price breaks below S1, price below 1w EMA50, volume above average
            short_cond = (close[i] < s1[i] and 
                         close[i] < ema50_1w_aligned[i] and
                         volume[i] > vol_ma20[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price closes below S1 OR price crosses below 1w EMA50
            if close[i] < s1[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price closes above R1 OR price crosses above 1w EMA50
            if close[i] > r1[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R1/S1 breakout with 1w EMA50 trend filter and volume confirmation.
# Works in bull markets via breakout continuation, in bear via mean reversion at S1/R1.
# Daily timeframe targets 7-25 trades/year to avoid fee drag. Volume filter ensures participation.
# Discrete sizing (0.25) minimizes churn. Works on BTC/ETH/SOL via institutional pivot levels.