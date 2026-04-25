#!/usr/bin/env python3
"""
1d Donchian(20) Breakout + Weekly EMA34 Trend + Volume Spike
Hypothesis: Daily Donchian channel breakouts capture medium-term momentum. 
Weekly EMA34 filters trend direction (long only when price above weekly EMA34 in bull/accumulation, 
short only when below in bear/distribution). Volume spike confirms institutional participation. 
Discrete sizing (0.30) balances profit potential and fee drag. 
Works in both bull/bear markets: in bull, longs on upper band breakouts with uptrend; 
in bear, shorts on lower band breakouts with downtrend. 
Target: 15-25 trades/year (60-100 over 4 years).
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
    
    # Get weekly data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate ATR for volatility filtering (using 14 periods)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for ATR (14) and Donchian (20)
    start_idx = max(14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1w_aligned[i]
        atr_value = atr[i]
        
        # Donchian channel: 20-period high/low (lookback excludes current bar)
        if i >= 20:
            donchian_high = np.max(high[i-20:i])
            donchian_low = np.min(low[i-20:i])
        else:
            donchian_high = np.max(high[:i])
            donchian_low = np.min(low[:i])
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Breakout conditions: price breaks above upper band or below lower band
        bullish_breakout = curr_close > donchian_high
        bearish_breakout = curr_close < donchian_low
        
        # Exit conditions: reverse breakout or trend rejection
        if position != 0:
            exit_signal = False
            
            if position == 1:
                # Exit on bearish breakout below lower band or trend rejection
                if bearish_breakout or curr_close < ema_trend:
                    exit_signal = True
                    
            elif position == -1:
                # Exit on bullish breakout above upper band or trend rejection
                if bullish_breakout or curr_close > ema_trend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions: Donchian breakout + trend alignment + volume spike
        if position == 0:
            # Long: break above upper band AND price above weekly EMA34
            long_condition = bullish_breakout and (curr_close > ema_trend) and volume_spike
            # Short: break below lower band AND price below weekly EMA34
            short_condition = bearish_breakout and (curr_close < ema_trend) and volume_spike
            
            if long_condition:
                signals[i] = 0.30
                position = 1
            elif short_condition:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            signals[i] = 0.30
        elif position == -1:
            signals[i] = -0.30
    
    return signals

name = "1d_Donchian20_Breakout_WeeklyEMA34_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0