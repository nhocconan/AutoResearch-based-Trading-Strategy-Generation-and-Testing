#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1dTrend_VolumeConfirm
Hypothesis: For BTC/ETH in bear/range markets (2025+), fading extreme Camarilla levels (R3/S3) with 1d trend filter and volume confirmation provides mean-reversion edge. Uses 6h timeframe to reduce trade frequency vs lower TFs. Target: 12-35 trades/year (50-140 over 4 years) to minimize fee drag. Works in bull markets via 1d trend filter avoiding counter-trend fades, and in bear markets via mean reversion at extremes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla calculation and trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # Prior 1d bar OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels: R3, S3 (extreme fade levels), R4, S4 (breakout continuation)
    camarilla_range = prev_high - prev_low
    r3 = prev_close + camarilla_range * 1.1 / 2
    s3 = prev_close - camarilla_range * 1.1 / 2
    r4 = prev_close + camarilla_range * 1.1
    s4 = prev_close - camarilla_range * 1.1
    
    # Align Camarilla levels to 6h timeframe (completed 1d bar)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # 1d EMA34 trend filter (loaded ONCE)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # ATR for stoploss calculation
    tr0 = np.abs(high - low)
    tr1 = np.abs(high[1:] - close[:-1])
    tr2 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[tr0[0]], np.maximum(tr1, tr2)])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1d EMA (34), volume MA (20), ATR (14)
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Look for entry signals
            # Long fade at S3: price < S3, volume spike, and above 1d EMA34 (bullish bias)
            long_fade = (curr_low < s3_aligned[i]) and volume_spike[i] and (curr_close > ema_34_1d_aligned[i])
            # Short fade at R3: price > R3, volume spike, and below 1d EMA34 (bearish bias)
            short_fade = (curr_high > r3_aligned[i]) and volume_spike[i] and (curr_close < ema_34_1d_aligned[i])
            
            if long_fade:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_fade:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price closes above R3 (fade failed) or trend reverses or ATR stoploss hit
            atr_stop = entry_price - 2.0 * atr[i]
            if curr_close > r3_aligned[i] or curr_close < ema_34_1d_aligned[i] or curr_close < atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price closes below S3 (fade failed) or trend reverses or ATR stoploss hit
            atr_stop = entry_price + 2.0 * atr[i]
            if curr_close < s3_aligned[i] or curr_close > ema_34_1d_aligned[i] or curr_close > atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0