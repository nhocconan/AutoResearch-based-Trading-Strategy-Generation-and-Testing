#!/usr/bin/env python3
name = "4h_Keltner_Breakout_Volume_Trend_v1"
timeframe = "4h"
leverage = 1.0

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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily EMA(20) for trend filter
    ema_20_1d = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Calculate 4h ATR(20) for Keltner channels
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 4h EMA(20) for Keltner middle line
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner bands
    upper = ema_20 + 2 * atr_20
    lower = ema_20 - 2 * atr_20
    
    # Volume spike detection: 4-period average
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Wait for EMA and ATR
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_1d_aligned[i]) or np.isnan(upper[i]) or 
            np.isnan(lower[i]) or np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: close above upper Keltner band with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_4[i] * 1.5
            daily_uptrend = ema_20_1d_aligned[i] > ema_20_1d_aligned[i-1]
            
            if close[i] > upper[i] and vol_condition and daily_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: close below lower Keltner band with volume and daily downtrend
            elif close[i] < lower[i] and vol_condition and not daily_uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: close back below EMA(20) or volume drops
            if close[i] < ema_20[i] or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: close back above EMA(20) or volume drops
            if close[i] > ema_20[i] or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Keltner channel breakout with volume confirmation and daily trend filter
# - Keltner channels (EMA20 ± 2*ATR) adapt to volatility better than fixed bands
# - Breakout above upper band with volume in daily uptrend = long opportunity
# - Breakdown below lower band with volume in daily downtrend = short opportunity
# - Volume confirmation (1.5x average) reduces false breakouts
# - Exit when price returns to middle line (EMA20) or volume weakens
# - Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend)
# - Position size 0.25 targets ~30-60 trades/year, avoiding fee drag
# - Daily trend filter ensures we trade with higher timeframe momentum
# - Volatility-adjusted channels prevent whipsaws in low volatility periods
# - Designed for BTC/ETH primary focus with potential applicability to SOL
# - Aims for 80-160 total trades over 4 years (20-40/year) to stay within limits