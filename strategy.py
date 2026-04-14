#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot mean reversion with daily trend filter and volume confirmation
# Targets: 12-37 trades/year by buying at S1 and selling at R1 in ranging markets
# Logic: Long when price touches S1 pivot in uptrend (price > daily EMA50) with volume confirmation
#        Short when price touches R1 pivot in downtrend (price < daily EMA50) with volume confirmation
#        Exit when price crosses the daily pivot point (PP) or trend reverses
# Position size: 0.25 to manage drawdown

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily high, low, close for Camarilla pivots
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for daily timeframe
    # PP = (H + L + C) / 3
    # S1 = C - (H - L) * 1.1 / 12
    # R1 = C + (H - L) * 1.1 / 12
    range_1d = high_1d - low_1d
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    s1_1d = close_1d - (range_1d * 1.1 / 12.0)
    r1_1d = close_1d + (range_1d * 1.1 / 12.0)
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # EMA10 for entry timing on 12h chart
    ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # ATR for dynamic thresholds
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align daily Camarilla levels and EMA50 to 12h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(20, n):
        if np.isnan(ema_10[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]):
            continue
        
        # Volume confirmation (1.3x average)
        volume_confirm = volume[i] > 1.3 * vol_ma[i]
        
        # Long: Price touches S1 in uptrend with volume confirmation
        if position == 0 and close[i] > ema_50_aligned[i] and close[i] <= s1_aligned[i] + 0.1 * atr[i] and volume_confirm:
            position = 1
            signals[i] = position_size
        # Short: Price touches R1 in downtrend with volume confirmation
        elif position == 0 and close[i] < ema_50_aligned[i] and close[i] >= r1_aligned[i] - 0.1 * atr[i] and volume_confirm:
            position = -1
            signals[i] = -position_size
        # Exit: Price crosses PP or trend reverses
        elif position != 0:
            if position == 1 and (close[i] < pp_aligned[i] or close[i] < ema_50_aligned[i]):
                position = 0
                signals[i] = 0.0
            elif position == -1 and (close[i] > pp_aligned[i] or close[i] > ema_50_aligned[i]):
                position = 0
                signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_Pivot_MeanReversion_TrendFilter"
timeframe = "12h"
leverage = 1.0