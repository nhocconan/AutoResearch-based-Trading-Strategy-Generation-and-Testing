#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: Use 12h timeframe with Camarilla R3/S3 breakout confirmed by 1d EMA34 trend and volume spike. Targets 12-37 trades/year to minimize fee drag. Works in bull/bear markets by using 1d EMA34 for trend direction and volume confirmation to filter false breakouts. Includes ATR-based stoploss to manage risk. Camarilla levels provide precise intraday support/resistance that adapts to volatility.
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
    
    # Calculate Camarilla levels (R3, S3) on 12h using previous bar's range
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    # Use previous bar to avoid look-ahead
    camarilla_r3 = np.zeros(n)
    camarilla_s3 = np.zeros(n)
    for i in range(1, n):
        if not (np.isnan(high[i-1]) or np.isnan(low[i-1]) or np.isnan(close[i-1])):
            camarilla_r3[i] = close[i-1] + 1.1 * (high[i-1] - low[i-1]) / 2.0
            camarilla_s3[i] = close[i-1] - 1.1 * (high[i-1] - low[i-1]) / 2.0
        else:
            camarilla_r3[i] = np.nan
            camarilla_s3[i] = np.nan
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
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
    
    # Warmup: need 2 for Camarilla (prev bar), 34 for 1d EMA, 20 for volume avg, 14 for ATR
    start_idx = max(2, 34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        atr_val = atr[i]
        size = 0.25  # 25% position size to balance risk and return
        
        if position == 0:
            # Flat - look for breakout with trend and volume confirmation
            # Long: break above Camarilla R3 + 1d EMA34 uptrend + volume spike
            long_entry = (close_val > camarilla_r3[i]) and \
                       (ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]) and \
                       volume_spike[i]
            # Short: break below Camarilla S3 + 1d EMA34 downtrend + volume spike
            short_entry = (close_val < camarilla_s3[i]) and \
                        (ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]) and \
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
            # Long - exit on Camarilla S3 break or ATR stoploss
            exit_condition = (close_val < camarilla_s3[i]) or \
                           (close_val < entry_price - 2.5 * atr_val)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on Camarilla R3 break or ATR stoploss
            exit_condition = (close_val > camarilla_r3[i]) or \
                           (close_val > entry_price + 2.5 * atr_val)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0