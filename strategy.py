#!/usr/bin/env python3
"""
1h Camarilla H3/L3 Breakout + 4h EMA50 Trend + Volume Spike + Session Filter
Hypothesis: Camarilla pivot levels (H3/L3) act as strong support/resistance on 1h. 
Breakouts above H3 or below L3 with volume confirmation and 4h EMA50 trend filter capture momentum moves. 
Session filter (08-20 UTC) reduces noise during low-liquidity hours. 
Uses 4h for signal direction (EMA50) and 1h for precise entry timing. 
Discrete sizing (0.20) limits fee drift. Target: 60-150 total trades over 4 years.
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
    open_time = prices['open_time']
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla pivot levels (H3, L3) from prior 1h bar (using 4h as proxy for stability)
    # For 1h strategy, we use prior 4h bar's Camarilla levels to avoid noise
    camarilla_h3_4h = df_4h['close'] + 1.1 * (df_4h['high'] - df_4h['low']) / 4
    camarilla_l3_4h = df_4h['close'] - 1.1 * (df_4h['high'] - df_4h['low']) / 4
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3_4h.values)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3_4h.values)
    
    # Calculate ATR for volatility filtering (using 14 periods on 1h)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for ATR (14) and EMA50 (50)
    start_idx = max(14, 50)
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any data not ready
        if np.isnan(ema_50_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_50_aligned[i]
        atr_value = atr[i]
        camarilla_h3_val = camarilla_h3_aligned[i]
        camarilla_l3_val = camarilla_l3_aligned[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Breakout conditions: price breaks above H3 or below L3
        bullish_breakout = curr_close > camarilla_h3_val
        bearish_breakout = curr_close < camarilla_l3_val
        
        # Exit conditions: reverse breakout or trend rejection
        if position != 0:
            exit_signal = False
            
            if position == 1:
                # Exit on bearish breakout below L3 or trend rejection
                if bearish_breakout or curr_close < ema_trend:
                    exit_signal = True
                    
            elif position == -1:
                # Exit on bullish breakout above H3 or trend rejection
                if bullish_breakout or curr_close > ema_trend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions: Camarilla breakout + trend alignment + volume spike
        if position == 0:
            # Long: break above H3 AND price above 4h EMA50
            long_condition = bullish_breakout and (curr_close > ema_trend) and volume_spike
            # Short: break below L3 AND price below 4h EMA50
            short_condition = bearish_breakout and (curr_close < ema_trend) and volume_spike
            
            if long_condition:
                signals[i] = 0.20
                position = 1
            elif short_condition:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            signals[i] = 0.20
        elif position == -1:
            signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA50_VolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0