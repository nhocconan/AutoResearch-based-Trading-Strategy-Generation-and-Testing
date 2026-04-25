#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_ATRStop
Hypothesis: On 12h timeframe, Camarilla R1/S1 breakouts with 1d EMA34 trend filter and volume spike (>1.8x 20-bar avg) captures institutional breakouts with low trade frequency. ATR-based stoploss (2.5x ATR) manages risk. Designed for 12-30 trades/year to minimize fee drag. Works in bull markets via long breakouts and bear markets via short breakouts.
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
    
    # Get 1d data for HTF trend and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA34 on 1d for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR (14) on 1d for stoploss
    tr_1d = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
    )
    tr_1d = np.concatenate([[np.nan], tr_1d])
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate Camarilla levels from previous 1d bar (R1, S1)
    # Camarilla: R1 = close + 1.1*(high-low)*1.0/12, S1 = close - 1.1*(high-low)*1.0/12
    # Use previous completed 1d bar to avoid look-ahead
    prev_close = np.concatenate([[np.nan], close_1d[:-1]])
    prev_high = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low = np.concatenate([[np.nan], low_1d[:-1]])
    
    camarilla_range = prev_high - prev_low
    r1 = prev_close + 1.1 * camarilla_range * 1.0 / 12
    s1 = prev_close - 1.1 * camarilla_range * 1.0 / 12
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume average (20-period) for volume spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    entry_bar = 0
    
    # Start index: need warmup for calculations
    start_idx = max(50, 20)  # EMA34, ATR, vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        ema_val = ema_34_aligned[i]
        atr_val = atr_14_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: current volume > 1.8x 20-period average
        volume_spike = vol_val > 1.8 * vol_ma_val
        
        if position == 0:
            # Look for entry signals: Camarilla R1/S1 breakout with trend and volume
            # Long: price breaks above R1 with uptrend (close > EMA34) and volume spike
            long_signal = (high_val > r1_val) and (close_val > ema_val) and volume_spike
            # Short: price breaks below S1 with downtrend (close < EMA34) and volume spike
            short_signal = (low_val < s1_val) and (close_val < ema_val) and volume_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                entry_bar = i
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                entry_bar = i
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. ATR-based stoploss: price drops below entry - 2.5 * ATR
            if close_val < entry_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                entry_bar = 0
            # 2. Opposite breakout: price breaks below S1 (exit long)
            elif close_val < s1_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                entry_bar = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. ATR-based stoploss: price rises above entry + 2.5 * ATR
            if close_val > entry_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                entry_bar = 0
            # 2. Opposite breakout: price breaks above R1 (exit short)
            elif close_val > r1_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                entry_bar = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_ATRStop"
timeframe = "12h"
leverage = 1.0