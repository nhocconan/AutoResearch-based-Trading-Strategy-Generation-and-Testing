#!/usr/bin/env python3
"""
1d_Williams_Alligator_Trend_JawTeethLips_1wTrendFilter
Hypothesis: Trade 1d Williams Alligator crossovers with 1w trend filter and volume confirmation.
Alligator consists of Jaw (13-period SMA shifted 8), Teeth (8-period SMA shifted 5), Lips (5-period SMA shifted 3).
Long when Lips > Teeth > Jaw AND price above Lips AND 1w uptrend AND volume spike.
Short when Lips < Teeth < Jaw AND price below Lips AND 1w downtrend AND volume spike.
Exit when Alligator reverses or ATR stoploss hit.
Target: 30-100 total trades over 4 years (7-25/year) for fee efficiency on 1d timeframe.
Works in both bull and bear markets by following the Alligator's alignment with higher timeframe trend.
"""

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
    
    # Get 1d data for Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 1d
    # Jaw: 13-period SMA shifted 8 bars
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMA shifted 5 bars
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMA shifted 3 bars
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Calculate 1w EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate ATR(14) for stoploss on 1d
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume spike filter: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of Alligator shifts (13+8=21), 1w EMA, volume MA, ATR
    start_idx = max(21, 34, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        # Alligator alignment: Lips > Teeth > Jaw for uptrend, Lips < Teeth < Jaw for downtrend
        alligator_long = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        alligator_short = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
        price_above_lips = close_val > lips[i]
        price_below_lips = close_val < lips[i]
        trend_1w_up = close_val > ema_34_1w_aligned[i]
        trend_1w_down = close_val < ema_34_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: Alligator aligned up AND price above lips AND 1w trend up AND volume spike
            long_signal = alligator_long and price_above_lips and trend_1w_up and vol_spike
            
            # Short: Alligator aligned down AND price below lips AND 1w trend down AND volume spike
            short_signal = alligator_short and price_below_lips and trend_1w_down and vol_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Alligator reverses down OR price hits ATR stoploss
            if (not alligator_long) or (close_val < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Alligator reverses up OR price hits ATR stoploss
            if (not alligator_short) or (close_val > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Williams_Alligator_Trend_JawTeethLips_1wTrendFilter"
timeframe = "1d"
leverage = 1.0