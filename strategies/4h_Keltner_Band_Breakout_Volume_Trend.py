#!/usr/bin/env python3
"""
4h_Keltner_Band_Breakout_Volume_Trend
Hypothesis: Price breaks above/below Keltner bands (ATR-based) with volume spike and daily EMA34 trend filter on 4h timeframe.
Uses 1d EMA34 for trend direction to filter breakouts in both bull/bear markets.
Target: 20-40 trades/year to minimize fee drift while capturing strong directional moves with proper risk control.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily EMA34 for trend filter (loaded once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Keltner bands: 20-period EMA ± 2*ATR(10)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[tr1[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    upper_keltner = ema_20 + 2 * atr_10
    lower_keltner = ema_20 - 2 * atr_10
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(35, 20)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = upper_keltner[i]
        lower = lower_keltner[i]
        ema34 = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above upper Keltner band with volume spike and uptrend (price > daily EMA34)
            if (price > upper and vol_spike and price > ema34):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Keltner band with volume spike and downtrend (price < daily EMA34)
            elif (price < lower and vol_spike and price < ema34):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price closes below daily EMA34 OR breaks below lower band (reversal)
            if price < ema34 or price < lower:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price closes above daily EMA34 OR breaks above upper band (reversal)
            if price > ema34 or price > upper:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Keltner_Band_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0