#!/usr/bin/env python3
"""
12h Donchian(20) Breakout + 1d EMA50 Trend + Volume Spike + ATR Stoploss
Hypothesis: Donchian channel breakouts on 12h timeframe capture institutional moves. 
Filtered by 1d EMA50 trend and volume spikes to avoid false breakouts. 
ATR-based stoploss limits drawdown. Designed for 12h to target 12-37 trades/year.
Works in bull/bear via trend filter - only takes breakouts in direction of 1d EMA50.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate ATR(14) on 1d for stoploss
    tr1 = np.maximum(df_1d['high'].values, np.roll(df_1d['close'].values, 1)) - np.minimum(df_1d['low'].values, np.roll(df_1d['close'].values, 1))
    tr1[0] = df_1d['high'].values[0] - df_1d['low'].values[0]  # first bar
    atr_1d = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate Donchian(20) on 12h for breakout signals
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    donchian_high = pd.Series(df_12h['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_12h['low']).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h (no extra delay needed as they're based on completed 12h bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Volume confirmation: current volume > 2.0 * 20-period average (stricter for fewer trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_stop = 0.0
    
    # Start index: need enough for calculations
    start_idx = max(20, 50)  # Donchian20, EMA50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or np.isnan(vol_ma[i])):
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
            # Look for entry signals - require ALL conditions: Donchian breakout + trend + volume
            # Long: price breaks above Donchian high AND bullish bias AND volume spike
            long_entry = (curr_high > donchian_high_aligned[i]) and bullish_bias and vol_spike
            # Short: price breaks below Donchian low AND bearish bias AND volume spike
            short_entry = (curr_low < donchian_low_aligned[i]) and bearish_bias and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_stop = entry_price - 2.5 * atr_1d_aligned[i]  # 2.5 ATR stop
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_stop = entry_price + 2.5 * atr_1d_aligned[i]  # 2.5 ATR stop
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price hits ATR stoploss OR Donchian low break (mean reversion) OR loss of bullish bias
            if (curr_low <= atr_stop) or (curr_low < donchian_low_aligned[i]) or (curr_close < ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price hits ATR stoploss OR Donchian high break (mean reversion) OR loss of bearish bias
            if (curr_high >= atr_stop) or (curr_high > donchian_high_aligned[i]) or (curr_close > ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dEMA50_Trend_VolumeSpike_ATRStop"
timeframe = "12h"
leverage = 1.0