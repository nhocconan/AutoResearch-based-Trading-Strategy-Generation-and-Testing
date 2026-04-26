#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_ATRStop
Hypothesis: 4h breakout above/below daily Camarilla R1/S1 levels in direction of 1d EMA34 trend, confirmed by volume spike (>2x 20-bar MA). Uses 1d HTF for trend alignment and Camarilla levels from daily OHLC for institutional support/resistance. Includes ATR-based stoploss to limit drawdown. Designed for 20-50 trades/year (80-200 total over 4 years) to avoid fee drag. Works in both bull and bear markets by following the 1d trend while using Camarilla structure for precise entries.
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
    
    # Load 1d data ONCE before loop for trend and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR for stoploss (using 1h data for responsiveness)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 30:
        return np.zeros(n)
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    tr1 = np.maximum(high_1h - low_1h, np.abs(high_1h - np.roll(close_1h, 1)), np.abs(low_1h - np.roll(close_1h, 1)))
    tr1[0] = high_1h[0] - low_1h[0]
    atr_1h = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    atr_1h_aligned = align_htf_to_ltf(prices, df_1h, atr_1h)
    
    # Calculate Camarilla levels from previous 1d bar (OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1, S1 levels (based on previous day's range)
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (no additional delay needed as they're based on completed 1d bar)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    base_size = 0.30  # Position size
    
    # Warmup: max of calculations (20 for vol, 34 for ema, 14 for atr)
    start_idx = max(20, 34, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma[i]) or
            np.isnan(atr_1h_aligned[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        ema_34_val = ema_34_1d_aligned[i]
        camarilla_r1_val = camarilla_r1_aligned[i]
        camarilla_s1_val = camarilla_s1_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr_1h_aligned[i]
        
        # Determine 1d trend: bullish if price > EMA34, bearish if price < EMA34
        bullish_1d = close_val > ema_34_val
        bearish_1d = close_val < ema_34_val
        
        # Entry conditions: breakout of Camarilla level in trend direction with volume
        long_entry = (close_val > camarilla_r1_val) and bullish_1d and vol_spike
        short_entry = (close_val < camarilla_s1_val) and bearish_1d and vol_spike
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Check stoploss: price < entry_price - 2.0 * ATR
            if entry_price > 0 and close_val < entry_price - 2.0 * atr_val:
                exit_long = True
            # Check trend reversal or opposite level touch
            elif (close_val < camarilla_s1_val) or not bullish_1d:
                exit_long = True
        elif position == -1:
            # Check stoploss: price > entry_price + 2.0 * ATR
            if entry_price > 0 and close_val > entry_price + 2.0 * atr_val:
                exit_short = True
            # Check trend reversal or opposite level touch
            elif (close_val > camarilla_r1_val) or not bearish_1d:
                exit_short = True
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0