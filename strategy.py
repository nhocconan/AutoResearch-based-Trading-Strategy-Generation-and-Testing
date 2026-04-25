#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike
Hypothesis: Camarilla R1/S1 breakout on 4h timeframe with 1d EMA34 trend filter and volume spike confirmation (>1.8x average volume). Only trade breakouts aligned with 1d EMA34 direction during high volume expansion. Uses discrete sizing (0.25) and ATR-based stoploss. Designed to capture strong momentum moves in both bull and bear markets by using the 1d EMA as a dynamic trend filter. Targets 19-50 trades/year on 4h timeframe to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla levels and ATR - primary timeframe
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 5:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate previous period's Camarilla levels (using prior 4h bar's HLC)
    # We need to shift the high/low/close by 1 to get the previous bar's values
    prev_high = np.roll(high_4h, 1)
    prev_low = np.roll(low_4h, 1)
    prev_close = np.roll(close_4h, 1)
    # Set first value to NaN since there's no previous bar
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla levels for R1, R2, S1, S2
    # R2 = Close + ((High - Low) * 1.1/2)
    # R1 = Close + ((High - Low) * 1.1/4)
    # S1 = Close - ((High - Low) * 1.1/4)
    # S2 = Close - ((High - Low) * 1.1/2)
    rang = prev_high - prev_low
    r2 = prev_close + (rang * 1.1 / 2)
    r1 = prev_close + (rang * 1.1 / 4)
    s1 = prev_close - (rang * 1.1 / 4)
    s2 = prev_close - (rang * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe
    r2_aligned = align_htf_to_ltf(prices, df_4h, r2)
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    s2_aligned = align_htf_to_ltf(prices, df_4h, s2)
    
    # Get 1d data for EMA34 trend filter - HTF
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA(34) on 1d
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA to 4h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate ATR(14) for stoploss on 4h
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr_4h = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # Calculate volume average (20-period) for volume spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(34, 20, 14)  # EMA needs 34, vol needs 20, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r2_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(s2_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or 
            np.isnan(atr_4h_aligned[i]) or 
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
        r2_val = r2_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        s2_val = s2_aligned[i]
        ema_val = ema_1d_aligned[i]
        atr_val = atr_4h_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: current volume > 1.8x 20-period average
        volume_spike = vol_val > 1.8 * vol_ma_val
        
        if position == 0:
            # Look for entry signals: Camarilla breakout with trend and volume confirmation
            # Long: price breaks above R1, above 1d EMA34, with volume spike
            long_signal = (high_val > r1_val) and (close_val > ema_val) and volume_spike
            # Short: price breaks below S1, below 1d EMA34, with volume spike
            short_signal = (low_val < s1_val) and (close_val < ema_val) and volume_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Stoploss: price moves against position by 2.0*ATR
            if close_val < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend reversal: price closes below 1d EMA34
            elif close_val < ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Stoploss: price moves against position by 2.0*ATR
            if close_val > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend reversal: price closes above 1d EMA34
            elif close_val > ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0