#!/usr/bin/env python3
name = "4h_Keltner_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # Load daily data ONCE before loop for trend filter and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily ATR(14)
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate daily EMA(20) for trend filter
    ema_20_1d = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Calculate Keltner Channels on 4h data
    ema_20_4h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_14_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    upper_keltner = ema_20_4h + (2 * atr_14_4h)
    lower_keltner = ema_20_4h - (2 * atr_14_4h)
    
    # Volume spike detection: 3-period average (0.75 days of 4h bars)
    vol_ma_3 = pd.Series(volume).rolling(window=3, min_periods=3).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 3)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_1d_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or 
            np.isnan(vol_ma_3[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above upper Keltner with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_3[i] * 2.0
            uptrend = ema_20_1d_aligned[i] > ema_20_1d_aligned[i-1]
            
            if close[i] > upper_keltner[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price below lower Keltner with volume and daily downtrend
            elif close[i] < lower_keltner[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below EMA(20) or volume drops
            if close[i] < ema_20_4h[i] or volume[i] < vol_ma_3[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above EMA(20) or volume drops
            if close[i] > ema_20_4h[i] or volume[i] < vol_ma_3[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Keltner Channel breakout with 1d trend and volume confirmation
# - Keltner Channels (EMA20 ± 2*ATR) adapt to volatility better than fixed bands
# - Breakout above upper Keltner with volume in daily uptrend = long opportunity
# - Breakdown below lower Keltner with volume in daily downtrend = short opportunity
# - Volume spike (2.0x average) confirms institutional participation
# - Daily EMA(20) trend filter reduces whipsaws and aligns with higher timeframe bias
# - Exit when price returns to EMA(20) middle line or volume weakens
# - Position size 0.25 targets ~25-50 trades/year to stay within limits
# - Works in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)
# - Volatility-adjusted bands prevent false signals during low volatility periods
# - Proven components: Keltner breakouts + volume confirmation + trend filtering
# - Avoids overtrading by requiring multiple confluence factors for entry