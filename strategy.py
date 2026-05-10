#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_Volume
Hypothesis: For 1h timeframe, use 4h trend (EMA50) as direction filter and 1h Camarilla R1/S1 breakouts for entry.
This combines higher timeframe trend alignment with lower timeframe precision entries to reduce whipsaw.
Volume confirmation ensures momentum behind breakouts. Designed for 15-30 trades/year to avoid fee drag.
Works in bull/bear by following 4h trend and requiring breakouts with volume.
"""

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

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
    
    # Get 4h data for EMA trend filter and Camarilla calculation (prior day's OHLC)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend direction
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels from prior day's OHLC (using 4h data approximated as daily)
    # Since we don't have daily, we'll use 4h to approximate: resample isn't allowed, so we use 4h close as proxy for daily close
    # Better: get actual 1d data for proper Camarilla
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Use prior day's OHLC for Camarilla calculation
    # We need to shift by 1 to use completed day's data
    prior_day_close = df_1d['close'].shift(1).values
    prior_day_high = df_1d['high'].shift(1).values
    prior_day_low = df_1d['low'].shift(1).values
    
    camarilla_r1 = prior_day_close + ((prior_day_high - prior_day_low) * 1.1 / 12)
    camarilla_s1 = prior_day_close - ((prior_day_high - prior_day_low) * 1.1 / 12)
    
    # Align Camarilla levels to 1h timeframe (use prior day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1, additional_delay_bars=1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1, additional_delay_bars=1)
    
    # Volume filter: volume > 1.5x 20-period average on 1h chart
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 4h EMA50
        price_above_ema = close[i] > ema_50_4h_aligned[i]
        price_below_ema = close[i] < ema_50_4h_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above R1 + above 4h EMA50 + volume spike
            if (close[i] > r1_aligned[i] and 
                price_above_ema and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below S1 + below 4h EMA50 + volume spike
            elif (close[i] < s1_aligned[i] and 
                  price_below_ema and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price breaks back below S1 (re-enters range) or volume drops below average
            if (close[i] < s1_aligned[i] or volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks back above R1 (re-enters range) or volume drops below average
            if (close[i] > r1_aligned[i] or volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals