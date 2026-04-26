#!/usr/bin/env python3
"""
1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike_v1
Hypothesis: Daily timeframe with Camarilla R3/S3 breakout confirmed by 1w EMA34 trend and volume spike. Targets 15-35 trades/year to minimize fee drag. Works in bull/bear markets by using 1w EMA34 for trend direction and volume confirmation to filter false breakouts. Includes ATR-based stoploss to manage risk.
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
    
    # Calculate Camarilla pivot levels (based on previous day)
    # R3 = close + 1.1 * (high - low) / 2
    # S3 = close - 1.1 * (high - low) / 2
    camarilla_high = close + 1.1 * (high - low) / 2
    camarilla_low = close - 1.1 * (high - low) / 2
    
    # Shift by 1 to use previous day's levels (no look-ahead)
    camarilla_high = np.roll(camarilla_high, 1)
    camarilla_low = np.roll(camarilla_low, 1)
    camarilla_high[0] = np.nan
    camarilla_low[0] = np.nan
    
    # Calculate 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    # ATR for stoploss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 1 for camarilla shift, 34 for 1w EMA, 20 for volume avg, 14 for ATR
    start_idx = max(1, 34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_high[i]) or np.isnan(camarilla_low[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_spike[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        atr_val = atr[i]
        size = 0.25  # 25% position size
        
        if position == 0:
            # Flat - look for breakout with trend and volume confirmation
            # Long: break above camarilla high + 1w EMA34 uptrend + volume spike
            long_entry = (close_val > camarilla_high[i]) and \
                       (ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1]) and \
                       volume_spike[i]
            # Short: break below camarilla low + 1w EMA34 downtrend + volume spike
            short_entry = (close_val < camarilla_low[i]) and \
                        (ema_34_1w_aligned[i] < ema_34_1w_aligned[i-1]) and \
                        volume_spike[i]
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on camarilla low break or ATR stoploss
            exit_condition = (close_val < camarilla_low[i]) or \
                           (close_val < entry_price - 2.5 * atr_val)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on camarilla high break or ATR stoploss
            exit_condition = (close_val > camarilla_high[i]) or \
                           (close_val > entry_price + 2.5 * atr_val)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0