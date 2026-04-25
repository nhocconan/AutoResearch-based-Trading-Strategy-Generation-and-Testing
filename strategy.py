#!/usr/bin/env python3
"""
12h Donchian(20) Breakout + 1d EMA50 Trend + Volume Spike + ATR Stop
Hypothesis: Donchian channel breakouts capture strong momentum in both bull and bear markets.
1d EMA50 filter ensures we trade with the higher timeframe trend, reducing false breakouts.
Volume spike confirms institutional participation. ATR-based stoploss manages risk.
Designed for 12h timeframe to target 12-37 trades/year (50-150 total over 4 years).
Works in bull markets (buy upper breakout in uptrend) and bear markets (sell lower breakdown in downtrend).
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
    
    # Load 1d data ONCE before loop for EMA50 and Donchian reference
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Daily Donchian channels (20-period) for breakout levels
    # Using prior day's close to avoid look-ahead (breakout of previous day's range)
    donchian_high = pd.Series(df_1d['close']).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(df_1d['close']).rolling(window=20, min_periods=20).min().shift(1).values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # ATR(14) for stoploss calculation
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_stop = 0.0
    
    # Start index: need enough for calculations
    start_idx = max(20, 50, 20, 14)  # Donchian, EMA, volume MA, ATR
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to 1d EMA50
        bullish_bias = curr_close > ema_1d_aligned[i]
        bearish_bias = curr_close < ema_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above Donchian upper band AND bullish bias AND volume spike
            long_entry = (curr_high > donchian_high_aligned[i]) and bullish_bias and vol_spike
            # Short: price breaks below Donchian lower band AND bearish bias AND volume spike
            short_entry = (curr_low < donchian_low_aligned[i]) and bearish_bias and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_stop = entry_price - 2.5 * atr[i]
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_stop = entry_price + 2.5 * atr[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Update trailing stop
            atr_stop = max(atr_stop, curr_close - 2.5 * atr[i])
            # Exit: price hits ATR stoploss OR loss of bullish bias
            if curr_low <= atr_stop or curr_close < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Update trailing stop
            atr_stop = min(atr_stop, curr_close + 2.5 * atr[i])
            # Exit: price hits ATR stoploss OR loss of bearish bias
            if curr_high >= atr_stop or curr_close > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dEMA50_Trend_VolumeSpike_ATRStop"
timeframe = "12h"
leverage = 1.0