#!/usr/bin/env python3
# 1h_4d_Triple_Momentum_Confluence
# Hypothesis: Combines 4h trend (EMA21), 1d momentum (ROC20), and 1h volume spike with time filtering (08-20 UTC) to capture high-probability moves.
# Works in bull/bear: Uses momentum for direction and volume for confirmation, avoiding chop.
# Target: 15-30 trades/year via strict 3-condition confluence.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4d_Triple_Momentum_Confluence"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # === 4h: EMA21 for trend ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    ema_4h = pd.Series(df_4h['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # === 1d: ROC20 for momentum ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    roc_1d = pd.Series(df_1d['close']).pct_change(periods=20).values
    roc_1d_aligned = align_htf_to_ltf(prices, df_1d, roc_1d)
    
    # === 1h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # === Time filter: 08-20 UTC ===
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(21, n):  # Start after EMA warmup
        # Skip outside trading hours
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        ema_val = ema_4h_aligned[i]
        roc_val = roc_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        close_val = prices['close'].iloc[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_val) or np.isnan(roc_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above EMA, positive ROC, volume spike
            if (close_val > ema_val and 
                roc_val > 0.02 and  # 2% monthly momentum
                vol_ratio_val > 1.8):
                signals[i] = 0.20
                position = 1
            # Short: Price below EMA, negative ROC, volume spike
            elif (close_val < ema_val and 
                  roc_val < -0.02 and  # -2% monthly momentum
                  vol_ratio_val > 1.8):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses below EMA or momentum turns negative
            if close_val < ema_val or roc_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: Price crosses above EMA or momentum turns positive
            if close_val > ema_val or roc_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals