#!/usr/bin/env python3
name = "6h_Keltner_Channel_1wTrend_Filter"
timeframe = "6h"
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
    
    # 1. Load 1w data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 2. 1w EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 3. Keltner Channel: EMA20 + ATR(10)*2
    ema20 = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr3 = np.absolute(high - np.roll(low, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    upper_keltner = ema20 + 2 * atr10
    lower_keltner = ema20 - 2 * atr10
    
    # 4. Volume filter: 20-period EMA for spike detection
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    # 5. Fixed position size to avoid churn
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(upper_keltner[i]) or 
            np.isnan(lower_keltner[i]) or np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        price_above_upper = close[i] > upper_keltner[i]
        price_below_lower = close[i] < lower_keltner[i]
        price_above_ema50 = close[i] > ema50_1w_aligned[i]
        price_below_ema50 = close[i] < ema50_1w_aligned[i]
        
        if position == 0:
            # Long: Price breaks above upper Keltner + above 1w EMA50 + volume spike
            if price_above_upper and price_above_ema50 and volume_ok[i]:
                signals[i] = position_size
                position = 1
            # Short: Price breaks below lower Keltner + below 1w EMA50 + volume spike
            elif price_below_lower and price_below_ema50 and volume_ok[i]:
                signals[i] = -position_size
                position = -1
        else:
            # Exit: Price crosses back through EMA20 (middle of Keltner)
            if position == 1:
                if close[i] < ema20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                if close[i] > ema20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals