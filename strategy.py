#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h KAMA + Volume Spike + Weekly Trend Filter
# Hypothesis: KAMA adapts to market noise, reducing false signals in choppy markets.
# Volume spike confirms institutional participation in KAMA direction breaks.
# Weekly trend filter ensures alignment with higher-timeframe momentum.
# Designed for 4h timeframe with low trade frequency (19-50/year).
# Works in bull via KAMA up + weekly uptrend + volume, in bear via KAMA down + weekly downtrend + volume.

name = "4h_kama_volume_weekly_trend_v1"
timeframe = "4h"
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Kaufman's Adaptive Moving Average (KAMA)
    def kama(close, er_length=10, fast_sc=2, slow_sc=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.abs(np.diff(close, prepend=close[0]))
        for i in range(1, len(close)):
            volatility[i] += volatility[i-1]
        volatility = np.concatenate([[0], np.diff(volatility, prepend=0)])
        
        er = np.zeros_like(close)
        for i in range(er_length, len(close)):
            price_change = np.abs(close[i] - close[i-er_length])
            sum_vol = volatility[i] - volatility[i-er_length] if i >= er_length else volatility[i]
            er[i] = price_change / (sum_vol + 1e-10)
        
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        kama_out = np.zeros_like(close)
        kama_out[0] = close[0]
        for i in range(1, len(close)):
            kama_out[i] = kama_out[i-1] + sc[i] * (close[i] - kama_out[i-1])
        return kama_out
    
    # KAMA(10,2,30)
    kama_vals = kama(close, 10, 2, 30)
    
    # Weekly trend filter: EMA(20) of weekly close
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(kama_vals[i]) or np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: KAMA turns down OR weekly trend turns bearish
            if kama_vals[i] < kama_vals[i-1] or close[i] < ema_20_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: KAMA turns up OR weekly trend turns bullish
            if kama_vals[i] > kama_vals[i-1] or close[i] > ema_20_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: KAMA turns up with weekly uptrend
                if kama_vals[i] > kama_vals[i-1] and close[i] > ema_20_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: KAMA turns down with weekly downtrend
                elif kama_vals[i] < kama_vals[i-1] and close[i] < ema_20_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals