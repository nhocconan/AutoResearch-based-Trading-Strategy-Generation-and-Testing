#!/usr/bin/env python3
name = "6h_ElderRay_BullBearPower_1dTrend"
timeframe = "6h"
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
    
    # 1d trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Elder Ray on 6h: Bull Power = High - EMA13, Bear Power = Low - EMA13
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough data for 1d EMA50
    
    for i in range(start_idx, n):
        # Skip if 1d trend data not ready
        if np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        bull_power = high[i] - ema13[i]
        bear_power = low[i] - ema13[i]
        
        if position == 0:
            # Long: Bull Power > 0 (strong buying pressure) + Bear Power < 0 (weak selling) + 1d uptrend
            if bull_power > 0 and bear_power < 0 and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (strong selling pressure) + Bull Power < 0 (weak buying) + 1d downtrend
            elif bear_power < 0 and bull_power < 0 and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when Bull Power <= 0 (buying pressure fades) or Bear Power > 0 (selling pressure emerges)
            if bull_power <= 0 or bear_power > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when Bear Power >= 0 (selling pressure fades) or Bull Power > 0 (buying pressure emerges)
            if bear_power >= 0 or bull_power > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals