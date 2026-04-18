#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h/12h momentum with volume surge and ATR volatility filter.
# Long when price breaks above 12h EMA(34) with volume > 1.8x 48-period average and ATR > 0.
# Short when price breaks below 12h EMA(34) with same conditions.
# Exit when price crosses back over 12h EMA(34).
# Uses 12h EMA for trend filter, volume surge for conviction, ATR for volatility.
# Designed for ~20-40 trades/year per symbol.
name = "4h_12hEMA34_VolumeSurge_ATR_Filter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # EMA(34) on 12h close
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # ATR(14) on 12h for volatility filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - close_12h)
    tr3 = np.abs(low_12h - close_12h)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Volume filter: current volume > 1.8 * 48-period average (48 * 4h = 8 days)
    vol_ma_48 = pd.Series(volume).rolling(window=48, min_periods=48).mean().values
    volume_filter = volume > (1.8 * vol_ma_48)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(atr_12h_aligned[i]) or
            np.isnan(vol_ma_48[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_val = ema_34_12h_aligned[i]
        atr_val = atr_12h_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: price above EMA with volume surge and volatility
            if close_val > ema_val and vol_filter and atr_val > 0:
                signals[i] = 0.25
                position = 1
            # Short: price below EMA with volume surge and volatility
            elif close_val < ema_val and vol_filter and atr_val > 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below EMA
            if close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above EMA
            if close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals