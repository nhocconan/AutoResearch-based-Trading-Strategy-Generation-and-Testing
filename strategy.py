#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
Long when price breaks above Camarilla R3 AND close > 1d EMA34 AND volume > 2.0x 20-period average.
Short when price breaks below Camarilla S3 AND close < 1d EMA34 AND volume > 2.0x 20-period average.
Exit when price retraces to Camarilla pivot point (PP) OR ATR trailing stop (2.5*ATR from extreme).
Uses discrete position sizing (0.30) targeting 20-50 trades/year on 4h timeframe.
Camarilla levels from daily timeframe provide robust support/resistance structure, proven on ETH/USDT.
4h timeframe balances noise reduction and trade frequency to avoid fee drag while capturing trends.
Volume confirmation at 2.0x average filters weak breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from daily timeframe (using previous day's data)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    rng = high_1d - low_1d
    pp = (high_1d + low_1d + close_1d) / 3.0
    r3 = pp + (high_1d - low_1d) * 1.1 / 4.0
    s3 = pp - (high_1d - low_1d) * 1.1 / 4.0
    
    # Align Camarilla levels to 4h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 14)  # EMA34 needs 34, vol MA needs 20, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        ema34_val = ema34_1d_aligned[i]
        pp_val = pp_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        
        if position == 0:
            # Long: Break above Camarilla R3 AND uptrend (price > EMA34) AND volume spike (2.0x avg)
            if close[i] > r3_val and close[i] > ema34_val and volume[i] > 2.0 * vol_ma_val:
                signals[i] = 0.30
                position = 1
                highest_since_entry = price
            # Short: Break below Camarilla S3 AND downtrend (price < EMA34) AND volume spike (2.0x avg)
            elif close[i] < s3_val and close[i] < ema34_val and volume[i] > 2.0 * vol_ma_val:
                signals[i] = -0.30
                position = -1
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retraces to Camarilla pivot point (PP)
            if position == 1 and close[i] <= pp_val:
                exit_signal = True
            elif position == -1 and close[i] >= pp_val:
                exit_signal = True
            
            # ATR-based trailing stop: 2.5 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.5 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "4H_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeConfirmation_PPExit_ATRTrailingStop"
timeframe = "4h"
leverage = 1.0