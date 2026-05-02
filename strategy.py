#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA50 trend filter, volume spike (>1.8x average), session filter (08-20 UTC), and chop regime filter (CHOP < 61.8)
# Uses 4h HTF for EMA50 trend filter to reduce noise and false breakouts on 1h.
# Volume confirmation ensures strong participation. Chop filter avoids ranging markets.
# Session filter reduces noise trades during low-activity hours.
# Discrete sizing 0.20 to minimize fee churn. Target: 60-150 trades over 4 years (15-37/year).
# Primary timeframe: 1h, HTF: 4h for EMA50.

name = "1h_Camarilla_H3_L3_Breakout_4hEMA50_Volume_Session_Chop"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Calculate Camarilla levels H3 and L3 from 1h timeframe (using prior bar)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    camarilla_h3 = prev_close + (prev_high - prev_low) * 1.1 / 6
    camarilla_l3 = prev_close - (prev_high - prev_low) * 1.1 / 6
    
    # 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # Choppiness Index regime filter (avoid ranging markets)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    highest_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    
    atr_safe = np.where(atr == 0, 1e-10, atr)
    chop = 100 * np.log10((highest_high - lowest_low) / (atr_safe * np.sqrt(atr_period))) / np.log10(atr_period)
    chop_regime = chop < 61.8  # True when trending (CHOP < 61.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 60
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(chop_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above H3 AND price > 4h EMA50 AND volume spike AND trending regime AND session
            if (close[i] > camarilla_h3[i] and 
                close[i] > ema_50_4h_aligned[i] and 
                volume_spike[i] and 
                chop_regime[i] and 
                session_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below L3 AND price < 4h EMA50 AND volume spike AND trending regime AND session
            elif (close[i] < camarilla_l3[i] and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume_spike[i] and 
                  chop_regime[i] and 
                  session_filter[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price drops below L3 OR price < 4h EMA50
            if close[i] < camarilla_l3[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: Price rises above H3 OR price > 4h EMA50
            if close[i] > camarilla_h3[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals