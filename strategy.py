#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + 1d EMA34 Trend + Volume Spike + ATR Stop
Hypothesis: Donchian channel breakouts capture strong momentum. 1d EMA34 filter ensures trades align with daily trend. Volume spike confirms institutional participation. ATR-based stoploss manages risk. Works in bull/bear: longs in uptrend, shorts in downtrend. Discrete sizing (0.30) limits fee drag. Target: 25-60 trades/year (100-240 over 4 years).
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR for volatility filtering and stoploss (using 20 periods)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for ATR (20) and Donchian (20)
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema_34_aligned[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_aligned[i]
        atr_value = atr[i]
        
        # Calculate Donchian channels (20-period) - using completed bars only
        if i >= 20:
            donchian_high = np.max(high[i-19:i+1])  # includes current bar
            donchian_low = np.min(low[i-19:i+1])
        else:
            donchian_high = np.max(high[:i+1])
            donchian_low = np.min(low[:i+1])
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Breakout conditions: price breaks above upper Donchian or below lower Donchian
        bullish_breakout = curr_close > donchian_high
        bearish_breakout = curr_close < donchian_low
        
        # ATR-based stoploss: exit if price moves against position by 2.5 * ATR
        if position != 0:
            exit_signal = False
            
            if position == 1:
                # Long: exit if price drops below entry_price - 2.5 * ATR
                if curr_close < entry_price - 2.5 * atr_value:
                    exit_signal = True
                # Also exit on bearish breakout of lower Donchian (trend reversal)
                elif bearish_breakout:
                    exit_signal = True
                    
            elif position == -1:
                # Short: exit if price rises above entry_price + 2.5 * ATR
                if curr_close > entry_price + 2.5 * atr_value:
                    exit_signal = True
                # Also exit on bullish breakout of upper Donchian (trend reversal)
                elif bullish_breakout:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # Entry conditions: Donchian breakout + trend alignment + volume spike
        if position == 0:
            # Long: break above upper Donchian AND price above 1d EMA34
            long_condition = bullish_breakout and (curr_close > ema_trend) and volume_spike
            # Short: break below lower Donchian AND price below 1d EMA34
            short_condition = bearish_breakout and (curr_close < ema_trend) and volume_spike
            
            if long_condition:
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
        elif position == 1:
            signals[i] = 0.30
        elif position == -1:
            signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA34_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0