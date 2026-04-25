#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_12hEMA34_VolumeConfirm_ATRStop
Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA34 trend filter and volume confirmation (>2.0x 24-bar avg). 
Enters long when price closes above R3 in uptrend, short when closes below S3 in downtrend. 
Exits on reverse break or ATR-based stoploss (2.5x ATR). Uses discrete sizing (0.25) to limit fee churn.
Designed for 4h timeframe with ~20-50 trades/year, works in bull/bear by following 12h trend filter.
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
    
    # 12h data for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 12h EMA34 for trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Daily data for Camarilla pivot calculation (yesterday's OHLC)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    open_1d = df_1d['open'].values
    
    # Calculate Camarilla levels for today using yesterday's OHLC
    # Camarilla: R4 = close + ((high - low) * 1.5/2), R3 = close + ((high - low) * 1.25/2), etc.
    # But we use the standard Camarilla formula based on previous day's range
    rng = (high_1d - low_1d)  # True range of previous day
    camarilla_r3 = close_1d + (rng * 1.1 / 4)   # R3 = C + (HL * 1.1/4)
    camarilla_s3 = close_1d - (rng * 1.1 / 4)   # S3 = C - (HL * 1.1/4)
    camarilla_r4 = close_1d + (rng * 1.5 / 4)   # R4 = C + (HL * 1.5/4)
    camarilla_s4 = close_1d - (rng * 1.5 / 4)   # S4 = C - (HL * 1.5/4)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume spike: current volume > 2.0x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # ATR for stoploss (20-period)
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.roll(close, 1))).values
    tr3 = pd.Series(np.abs(low - np.roll(close, 1))).values
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_stop = 0.0
    
    # Start index: need enough data for ATR and indicators
    start_idx = max(24, 20)  # volume MA and ATR
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(atr[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for long entry: price closes above R3 in uptrend with volume confirmation
            long_breakout = (curr_close > camarilla_r3_aligned[i]) and \
                           (ema_34_12h_aligned[i] > camarilla_r3_aligned[i]) and \
                           volume_spike[i]
                           
            # Check for short entry: price closes below S3 in downtrend with volume confirmation
            short_breakout = (curr_close < camarilla_s3_aligned[i]) and \
                            (ema_34_12h_aligned[i] < camarilla_s3_aligned[i]) and \
                            volume_spike[i]
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_stop = entry_price - (2.5 * atr[i])
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_stop = entry_price + (2.5 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long position
            signals[i] = 0.25
            # Update trailing stop (only move up)
            atr_stop = max(atr_stop, curr_close - (2.5 * atr[i]))
            # Exit: price closes below S3 (reverse break) OR stoploss hit
            if (curr_close < camarilla_s3_aligned[i]) or (curr_low <= atr_stop):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short position
            signals[i] = -0.25
            # Update trailing stop (only move down)
            atr_stop = min(atr_stop, curr_close + (2.5 * atr[i]))
            # Exit: price closes above R3 (reverse break) OR stoploss hit
            if (curr_close > camarilla_r3_aligned[i]) or (curr_high >= atr_stop):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_12hEMA34_VolumeConfirm_ATRStop"
timeframe = "4h"
leverage = 1.0