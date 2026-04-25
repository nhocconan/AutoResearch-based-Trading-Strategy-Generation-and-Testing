#!/usr/bin/env python3
"""
4h Camarilla R3S3 Breakout with 12h EMA34 Trend and Volume Spike
Hypothesis: Camarilla pivot levels (R3/S3) act as strong intraday support/resistance.
Breakout above R3 or below S3 with 12h EMA34 trend alignment and volume confirmation
captures institutional momentum moves. Works in bull markets (breakouts above R3 in uptrend)
and bear markets (breakouts below S3 in downtrend). 4h timeframe targets 20-50 trades/year.
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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA34 for trend filter
    ema_34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(20, 34)  # volume MA, 12h EMA alignment
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla levels for current 4h bar using prior bar's OHLC
        if i == 0:
            continue
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        rang = prev_high - prev_low
        
        # Camarilla R3 and S3 levels
        r3 = prev_close + rang * 1.1 / 4
        s3 = prev_close - rang * 1.1 / 4
        
        curr_close = close[i]
        curr_volume = volume[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to 12h EMA34
        uptrend = ema_34_aligned[i] is not None and curr_close > ema_34_aligned[i]
        downtrend = ema_34_aligned[i] is not None and curr_close < ema_34_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: break above R3 AND uptrend AND volume spike
            long_entry = (curr_close > r3) and uptrend and vol_spike
            # Short: break below S3 AND downtrend AND volume spike
            short_entry = (curr_close < s3) and downtrend and vol_spike
            
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
            # Exit: price breaks below S3 (reversal) OR loss of uptrend
            if (curr_close < s3) or (curr_close < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price breaks above R3 (reversal) OR loss of downtrend
            if (curr_close > r3) or (curr_close > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_12hEMA34_Trend_VolumeSp"
timeframe = "4h"
leverage = 1.0