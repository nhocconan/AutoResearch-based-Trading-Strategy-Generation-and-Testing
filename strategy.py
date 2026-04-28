#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R(14) extreme reversal + 1d EMA(34) trend filter + volume spike confirmation
# Williams %R identifies overbought/oversold conditions; reversals from extremes work well in ranging/choppy markets.
# 1d EMA(34) ensures alignment with daily trend to avoid counter-trend trades.
# Volume spike (>2x 24-bar average) adds conviction to reversals.
# Designed for 6h timeframe targeting 12-37 trades/year with discrete sizing (0.25) to minimize fee drag.

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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    
    # Align 1d EMA to 6h (changes only when 1d bar closes)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).values
    
    # Volume confirmation: >2.0x 24-bar average volume (4d equivalent on 6h)
    volume_series = pd.Series(volume)
    volume_ma_24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > 2.0 * volume_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(14, 24, 34)  # Williams %R(14), volume MA(24), 1d EMA(34)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_ma_24[i])):
            signals[i] = 0.0
            continue
        
        wr = williams_r[i]
        vol_spike = volume_spike[i]
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for reversal entries from extremes
            # Long entry: Williams %R crosses above -80 from oversold, above 1d EMA, volume spike
            if i > start_idx and williams_r[i-1] <= -80 and wr > -80 and price > ema_34_1d_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: Williams %R crosses below -20 from overbought, below 1d EMA, volume spike
            elif i > start_idx and williams_r[i-1] >= -20 and wr < -20 and price < ema_34_1d_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on reversal signal or trend violation
            # Exit: Williams %R crosses below -50 (momentum loss) OR price below 1d EMA
            if i > start_idx and (williams_r[i-1] >= -50 and wr < -50) or price < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on reversal signal or trend violation
            # Exit: Williams %R crosses above -50 (momentum loss) OR price above 1d EMA
            if i > start_idx and (williams_r[i-1] <= -50 and wr > -50) or price > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals