#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla H4/L4 breakout with 1d EMA50 trend filter, volume spike (>1.8x average), and chop regime filter (CHOP > 61.8)
# Uses 1d HTF for Camarilla levels and EMA50 to capture institutional structure and reduce false breakouts.
# Volume confirmation at 1.8x average ensures strong participation. Chop filter ensures trades only in ranging markets (mean reversion).
# Discrete sizing 0.25 to minimize fee churn. Target: 50-150 trades over 4 years (12-37/year).
# Primary timeframe: 12h, HTF: 1d for Camarilla levels and EMA50.

name = "12h_Camarilla_H4_L4_Breakout_1dEMA50_Volume_Chop"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels H4 and L4 from 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior 1d bar's high, low, close for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla H4 and L4 levels
    camarilla_h4 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_l4 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 1.8x 30-period average (balanced to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # Choppiness Index regime filter (CHOP > 61.8 = ranging market)
    chop_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).rolling(window=chop_period, min_periods=chop_period).mean().values
    highest_high = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    lowest_low = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    
    chop = 100 * np.log10(atr * np.sqrt(chop_period) / (highest_high - lowest_low + 1e-10)) / np.log10(chop_period)
    chop_regime = chop > 61.8  # True when ranging (CHOP > 61.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 80
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(chop_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above H4 AND price < 1d EMA50 (mean reversion in ranging market) AND volume spike AND chop regime
            if (close[i] > camarilla_h4_aligned[i] and 
                close[i] < ema_50_1d_aligned[i] and 
                volume_spike[i] and 
                chop_regime[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below L4 AND price > 1d EMA50 (mean reversion in ranging market) AND volume spike AND chop regime
            elif (close[i] < camarilla_l4_aligned[i] and 
                  close[i] > ema_50_1d_aligned[i] and 
                  volume_spike[i] and 
                  chop_regime[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price drops below L4 OR price > 1d EMA50
            if close[i] < camarilla_l4_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price rises above H4 OR price < 1d EMA50
            if close[i] > camarilla_h4_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals