#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla H3/L3 breakout with 1w EMA34 trend filter, volume spike (>2.0x average), and chop regime filter (CHOP < 61.8)
# Uses H3/L3 for strong intraday reversal levels. Volume threshold reduces churn. Discrete sizing 0.25.
# Target: 30-100 trades over 4 years. Works in bull/bear via trend filter and regime avoidance of chop.
# Primary timeframe: 1d, HTF: 1w for EMA34.

name = "1d_Camarilla_H3_L3_Breakout_1wEMA34_Volume_Chop"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels H3 and L3 from 1d timeframe (using prior day's data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior day's high, low, close for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla H3 and L3 levels
    camarilla_h3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_l3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align Camarilla levels to 1d timeframe (they update daily)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: 2.0x 20-period average (stricter to avoid overtrading)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Choppiness Index regime filter (avoid ranging markets)
    # CHOP > 61.8 = ranging (avoid), CHOP < 38.2 = trending (favor)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    highest_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    
    # Avoid division by zero
    atr_safe = np.where(atr == 0, 1e-10, atr)
    chop = 100 * np.log10((highest_high - lowest_low) / (atr_safe * np.sqrt(atr_period))) / np.log10(atr_period)
    chop_regime = chop < 61.8  # True when trending (CHOP < 61.8), False when ranging
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(chop_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above H3 AND price > 1w EMA34 AND volume spike AND trending regime
            if (close[i] > camarilla_h3_aligned[i] and 
                close[i] > ema_34_1w_aligned[i] and 
                volume_spike[i] and 
                chop_regime[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below L3 AND price < 1w EMA34 AND volume spike AND trending regime
            elif (close[i] < camarilla_l3_aligned[i] and 
                  close[i] < ema_34_1w_aligned[i] and 
                  volume_spike[i] and 
                  chop_regime[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price drops below L3 OR price < 1w EMA34
            if close[i] < camarilla_l3_aligned[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price rises above H3 OR price > 1w EMA34
            if close[i] > camarilla_h3_aligned[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals