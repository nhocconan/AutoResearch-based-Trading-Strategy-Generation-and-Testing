#!/usr/bin/env python3
"""
12h CAMARILLA PIVOT + 1D TREND + VOLUME SPIKE
Hypothesis: CAMARILLA pivot levels on 12h provide strong intraday support/resistance.
Breakouts above/below key levels (H3/L3) with 1d trend alignment and volume spikes
capture momentum moves while avoiding false breakouts. Works in bull (breakouts above H3)
and bear (breakdowns below L3). Target: 12-37 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for CAMARILLA pivot calculation (based on prior day)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for CAMARILLA calculation
    prev_high = high_1d
    prev_low = low_1d
    prev_close = close_1d
    
    # CAMARILLA pivot levels
    # Pivot = (H + L + C) / 3
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    # Resistance levels
    r1 = pivot + (range_val * 1.0833)
    r2 = pivot + (range_val * 1.1666)
    r3 = pivot + (range_val * 1.2500)
    r4 = pivot + (range_val * 1.5000)
    
    # Support levels
    s1 = pivot - (range_val * 1.0833)
    s2 = pivot - (range_val * 1.1666)
    s3 = pivot - (range_val * 1.2500)
    s4 = pivot - (range_val * 1.5000)
    
    # Align CAMARILLA levels to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # ATR for volatility filter (14-period)
    tr1 = pd.Series(high).subtract(pd.Series(low)).abs()
    tr2 = pd.Series(high).subtract(pd.Series(close).shift(1)).abs()
    tr3 = pd.Series(low).subtract(pd.Series(close).shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Volatility filter: ATR > 20-period ATR mean (avoid choppy markets)
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    vol_filter = atr > atr_ma
    
    # Volume filter: current volume > 2.0x 24-period average (stricter)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_spike[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below S3 or trend reverses
            if close[i] < s3_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above R3 or trend reverses
            if close[i] > r3_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Trend filter: price vs 1d EMA50
            uptrend = close[i] > ema_50_1d_aligned[i]
            downtrend = close[i] < ema_50_1d_aligned[i]
            
            # Long: price breaks above R3 + uptrend + volume spike + vol filter
            if (close[i] > r3_aligned[i] and 
                uptrend and 
                vol_spike[i] and
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below S3 + downtrend + volume spike + vol filter
            elif (close[i] < s3_aligned[i] and 
                  downtrend and 
                  vol_spike[i] and
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals