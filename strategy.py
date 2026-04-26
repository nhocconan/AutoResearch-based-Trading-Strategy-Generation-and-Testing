#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike_v1
Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter, volume spike confirmation, and ATR-based stoploss. Targets 20-40 trades/year by requiring strong confluence of price structure, trend, and volume. Works in bull/bear markets via trend filter and avoids choppy regimes by focusing on strong breakouts.
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
    
    # Calculate Camarilla levels from previous 4h bar (using 4h HTF for daily-like structure)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Previous 4h bar's OHLC for Camarilla calculation
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    prev_close = df_4h['close'].shift(1).values
    
    # Camarilla levels: R3, S3, PP (pivot point)
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    camarilla_pp = (prev_high + prev_low + prev_close) / 3
    
    # Align to 4h timeframe (wait for completed 4h bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pp)
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    # ATR for stoploss
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 20 for volume avg, 50 for 12h EMA, 14 for ATR
    start_idx = max(20, 50, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_pp_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        atr_val = atr[i]
        size = 0.25  # 25% position size to manage drawdown
        
        if position == 0:
            # Flat - look for breakout with trend and volume confirmation
            # Long: break above R3 + 12h EMA50 uptrend + volume spike
            long_entry = (close_val > camarilla_r3_aligned[i]) and \
                       (ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]) and \
                       volume_spike[i]
            # Short: break below S3 + 12h EMA50 downtrend + volume spike
            short_entry = (close_val < camarilla_s3_aligned[i]) and \
                        (ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1]) and \
                        volume_spike[i]
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close[i+1] if i+1 < n else close_val  # entry at next bar open
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close[i+1] if i+1 < n else close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit conditions
            exit_signal = False
            # Stoploss: 2 * ATR below entry
            if close_val < entry_price - 2.0 * atr_val:
                exit_signal = True
            # Take profit: revert to pivot point
            elif close_val < camarilla_pp_aligned[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit conditions
            exit_signal = False
            # Stoploss: 2 * ATR above entry
            if close_val > entry_price + 2.0 * atr_val:
                exit_signal = True
            # Take profit: revert to pivot point
            elif close_val > camarilla_pp_aligned[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0