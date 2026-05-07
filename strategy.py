#!/usr/bin/env python3
name = "6h_Keltner_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Keltner Channel on 6h: EMA(20) +/- 2 * ATR(10)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    kc_upper = ema_20 + 2 * atr_10
    kc_lower = ema_20 - 2 * atr_10
    
    # Volume spike detection: 4-period average (1 day of 6h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 10, 4)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_20[i]) or 
            np.isnan(atr_10[i]) or np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or 
            np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Keltner upper with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_4[i] * 1.8
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if close[i] > kc_upper[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Keltner lower with volume and daily downtrend
            elif close[i] < kc_lower[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below EMA(20) or volume drops
            if close[i] < ema_20[i] or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above EMA(20) or volume drops
            if close[i] > ema_20[i] or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Keltner Channel breakout with 1d trend and volume confirmation
# - Keltner Channel (EMA20 ± 2*ATR10) adapts to volatility, better than fixed bands
# - Breakout above upper band with volume in daily uptrend = long opportunity
# - Breakdown below lower band with volume in daily downtrend = short opportunity
# - Volume spike (1.8x average) confirms institutional participation
# - Works in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)
# - Exit when price returns to EMA(20) or volume weakens
# - Position size 0.25 targets ~20-50 trades/year, avoiding fee drag
# - Uses daily trend filter to avoid whipsaws in ranging markets
# - Novel combination: Keltner breakout (6h) + trend (1d) + volume (6h) not recently tried
# - Aims for 50-150 total trades over 4 years (12-37/year) to stay within limits