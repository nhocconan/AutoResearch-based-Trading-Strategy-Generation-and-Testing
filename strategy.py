#!/usr/bin/env python3
"""
4h Camarilla R3/S3 Breakout + 1d EMA34 Trend + Volume Spike
Hypothesis: Camarilla pivot levels (R3/S3) act as strong support/resistance on 1d timeframe.
Breakouts above R3 or below S3 with 1d EMA34 trend alignment and volume confirmation
capture strong momentum moves. Works in bull/bear via higher timeframe trend filter.
Target: 20-40 trades/year on 4h to avoid fee drag.
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
    
    # Load 1d data ONCE before loop for pivot calculation and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # Typical price = (high + low + close) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_1d = df_1d['high'] - df_1d['low']
    
    # Camarilla levels: R3/S3 are the strongest breakout levels
    # R3 = close + (high - low) * 1.1 / 4
    # S3 = close - (high - low) * 1.1 / 4
    r3 = df_1d['close'] + range_1d * 1.1 / 4
    s3 = df_1d['close'] - range_1d * 1.1 / 4
    
    # Align the pivot levels to LTF (they represent the previous day's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3.values)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA34 and volume MA
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to 1d EMA34
        bullish_bias = curr_close > ema_1d_aligned[i]
        bearish_bias = curr_close < ema_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals - require: breakout + trend + volume
            # Long: break above R3 AND bullish bias AND volume spike
            long_entry = (curr_high > r3_aligned[i]) and bullish_bias and vol_spike
            # Short: break below S3 AND bearish bias AND volume spike
            short_entry = (curr_low < s3_aligned[i]) and bearish_bias and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below EMA34 (trend change) OR re-enters Camarilla H3/L3 range
            h3 = df_1d['close'].iloc[-1] + (df_1d['high'].iloc[-1] - df_1d['low'].iloc[-1]) * 1.1 / 6 if len(df_1d) > 0 else r3_aligned[i] * 0.95  # fallback
            l3 = df_1d['close'].iloc[-1] - (df_1d['high'].iloc[-1] - df_1d['low'].iloc[-1]) * 1.1 / 6 if len(df_1d) > 0 else s3_aligned[i] * 1.05  # fallback
            h3_aligned = align_htf_to_ltf(prices, df_1d, np.full(len(df_1d), h3)) if len(df_1d) > 0 else r3_aligned[i] * 0.95
            l3_aligned = align_htf_to_ltf(prices, df_1d, np.full(len(df_1d), l3)) if len(df_1d) > 0 else s3_aligned[i] * 1.05
            
            if (curr_close < ema_1d_aligned[i]) or (curr_low < h3_aligned[i] and curr_high > l3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above EMA34 (trend change) OR re-enters Camarilla H3/L3 range
            if (curr_close > ema_1d_aligned[i]) or (curr_low < h3_aligned[i] and curr_high > l3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0