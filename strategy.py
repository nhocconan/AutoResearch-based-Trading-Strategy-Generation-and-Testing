#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme reversal with 1d EMA34 trend filter and volume spike confirmation
# Williams %R(14) identifies oversold (< -80) and overbought (> -20) conditions.
# Long when %R crosses above -80 from below with 1d EMA34 uptrend and volume spike.
# Short when %R crosses below -20 from above with 1d EMA34 downtrend and volume spike.
# Uses 6h primary timeframe to balance trade frequency and capture multi-day mean reversion in bear markets.
# Target 12-37 trades/year via tight Williams %R extreme conditions.

name = "6h_WilliamsR14_ExtremeReversal_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Williams %R(14) on 6h data
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    # Avoid division by zero
    rr = highest_high - lowest_low
    rr[rr == 0] = 1e-10
    williams_r = -100 * (highest_high - close) / rr
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 20, 34)  # Williams %R, volume MA20, and EMA34 need sufficient history
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume_spike[i]
        wr = williams_r[i]
        ema34_val = ema34_1d_aligned[i]
        
        # Williams %R extreme reversal logic
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R crosses above -80 from below AND 1d EMA34 uptrend AND volume spike
            if i > start_idx and williams_r[i-1] <= -80 and wr > -80 and ema34_val > close[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R crosses below -20 from above AND 1d EMA34 downtrend AND volume spike
            elif i > start_idx and williams_r[i-1] >= -20 and wr < -20 and ema34_val < close[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when Williams %R rises above -20 (overbought) or reverse signal
            # Exit long when Williams %R rises above -20 (overbought territory) or reverse signal
            if wr >= -20 or (i > start_idx and williams_r[i-1] >= -20 and wr < -20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when Williams %R falls below -80 (oversold) or reverse signal
            # Exit short when Williams %R falls below -80 (oversold territory) or reverse signal
            if wr <= -80 or (i > start_idx and williams_r[i-1] <= -80 and wr > -80):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals