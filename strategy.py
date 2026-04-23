#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike.
Long when price breaks above R3 AND close > 1d EMA34 AND volume > 2.0x 20-period average.
Short when price breaks below S3 AND close < 1d EMA34 AND volume > 2.0x 20-period average.
Exit when price retouches the Camarilla pivot (PP) or ATR trailing stop (2.5*ATR from extreme).
Uses discrete position sizing (0.25) to minimize fee drag. Targets 12-30 trades/year per symbol.
Camarilla levels provide precise intraday support/resistance; EMA34 filters trend; volume confirms breakout strength.
Works in bull markets (breakouts continue) and bear markets (breakdowns continue) with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter (MTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 12h Camarilla levels (PP, R3, S3) from prior 12h bar
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Camarilla calculation: based on previous day's high, low, close
    # For 12h timeframe, we use the prior 12h bar's HLC
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate pivot point (PP) = (H + L + C) / 3
    pp_12h = (high_12h + low_12h + close_12h) / 3.0
    # R3 = PP + (H - L) * 1.1 / 2
    r3_12h = pp_12h + (high_12h - low_12h) * 1.1 / 2.0
    # S3 = PP - (H - L) * 1.1 / 2
    s3_12h = pp_12h - (high_12h - low_12h) * 1.1 / 2.0
    
    # Align to 12h timeframe (already in 12h bars, need to align to lower timeframe)
    pp_12h_aligned = align_htf_to_ltf(prices, df_12h, pp_12h)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    
    # Volume average (20-period) on 12h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA34 needs 34, vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(pp_12h_aligned[i]) or 
            np.isnan(r3_12h_aligned[i]) or np.isnan(s3_12h_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        ema34_val = ema34_1d_aligned[i]
        pp_val = pp_12h_aligned[i]
        r3_val = r3_12h_aligned[i]
        s3_val = s3_12h_aligned[i]
        
        if position == 0:
            # Long: Price breaks above R3 AND uptrend (close > EMA34) AND volume spike
            if close[i] > r3_val and close[i] > ema34_val and volume[i] > 2.0 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Price breaks below S3 AND downtrend (close < EMA34) AND volume spike
            elif close[i] < s3_val and close[i] < ema34_val and volume[i] > 2.0 * vol_ma_val:
                signals[i] = -0.25
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
            
            # Primary exit: Price retouches pivot point (PP)
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
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_ATRTrailingStop_PPExit"
timeframe = "12h"
leverage = 1.0