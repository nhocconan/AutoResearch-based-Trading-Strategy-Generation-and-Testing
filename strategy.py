#!/usr/bin/env python3
"""
1d Camarilla H3/L3 Breakout + 1w EMA50 Trend + Volume Spike
Hypothesis: On daily timeframe, Camarilla H3/L3 levels from prior week act as strong support/resistance.
Breakouts with 1-week EMA50 trend filter capture momentum in both bull/bear markets (long in uptrend, short in downtrend).
Volume spike confirms institutional participation. Weekly HTF avoids overtrading while capturing major moves.
Target: 7-25 trades/year (30-100 over 4 years) with discrete sizing (0.25) to minimize fee drag.
Works in bull markets via breakout longs and bear markets via breakdown shorts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data for Camarilla levels (prior week's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate prior week's Camarilla H3/L3 using previous Friday's OHLC
    # Need at least 5 trading days for a week
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Get Friday's OHLC (5 days ago from current daily bar)
    friday_high = df_1d['high'].shift(5).values
    friday_low = df_1d['low'].shift(5).values
    friday_close = df_1d['close'].shift(5).values
    
    # Calculate Camarilla levels for prior week (using Friday's data as weekly close)
    camarilla_h3 = friday_close + (friday_high - friday_low) * 1.1 / 4
    camarilla_l3 = friday_close - (friday_high - friday_low) * 1.1 / 4
    
    # Align Camarilla levels to 1d timeframe (wait for weekly close)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3, additional_delay_bars=0)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3, additional_delay_bars=0)
    
    # Calculate ATR for volatility (20-period on 1d)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA50 (50) and weekly alignment
    start_idx = 55  # 50 for EMA + 5 for weekly shift
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema_50_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_trend = ema_50_aligned[i]
        atr_value = atr[i]
        h3_level = camarilla_h3_aligned[i]
        l3_level = camarilla_l3_aligned[i]
        
        # Volume spike: current volume > 2.5 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.5 * vol_ma_20
        
        # Breakout conditions: price breaks above H3 or below L3
        bullish_breakout = curr_close > h3_level
        bearish_breakout = curr_close < l3_level
        
        # Exit conditions: reverse breakout or trend rejection
        if position != 0:
            exit_signal = False
            
            if position == 1:
                # Exit on bearish breakout below L3 or trend rejection (price below EMA)
                if bearish_breakout or curr_close < ema_trend:
                    exit_signal = True
                    
            elif position == -1:
                # Exit on bullish breakout above H3 or trend rejection (price above EMA)
                if bullish_breakout or curr_close > ema_trend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions: Camarilla breakout + trend alignment + volume spike
        if position == 0:
            # Long: break above H3 AND price above 1w EMA50
            long_condition = bullish_breakout and (curr_close > ema_trend) and volume_spike
            # Short: break below L3 AND price below 1w EMA50
            short_condition = bearish_breakout and (curr_close < ema_trend) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_H3L3_Breakout_1wEMA50_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0