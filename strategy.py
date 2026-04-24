#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R + 1d EMA34 trend filter with volume spike confirmation.
- Primary timeframe: 6h for entries/exits.
- HTF: 1d EMA34 for trend direction (bullish if price > EMA34, bearish if price < EMA34).
- Williams %R(14) for mean reversion: long when < -80 (oversold), short when > -20 (overbought).
- Volume: Current 6h volume > 1.8 * 20-period 1d volume MA to confirm participation.
- Entry: Long when Williams %R < -80 AND 1d EMA34 bullish (price > EMA) AND volume spike.
         Short when Williams %R > -20 AND 1d EMA34 bearish (price < EMA) AND volume spike.
- Exit: Williams %R crosses above -50 for longs, below -50 for shorts (mean reversion exit).
        Also exit on loss of volume confirmation or trend reversal.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
Williams %R captures short-term exhaustion in both bull and bear markets. Combined with
1d trend filter and volume confirmation, it avoids counter-trend trades and works in
ranging conditions by fading extremes only when aligned with higher timeframe momentum.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R(14) on 6h
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low + 1e-10) * -100
    
    # Get 1d data for EMA34 trend filter and volume MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    df_1d_close = df_1d['close'].values
    ema_1d = pd.Series(df_1d_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period volume MA on 1d
    df_1d_volume = df_1d['volume'].values
    vol_ma_1d = pd.Series(df_1d_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 6h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Volume confirmation: current 6h volume > 1.8 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (1.8 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 14)  # Need enough bars for EMA34, volume MA, and Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or
            np.isnan(williams_r[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        ema_val = ema_1d_aligned[i]
        wr = williams_r[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish: Williams %R oversold (< -80) AND 1d EMA34 bullish (price > EMA)
                if wr < -80 and curr_close > ema_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish: Williams %R overbought (> -20) AND 1d EMA34 bearish (price < EMA)
                elif wr > -20 and curr_close < ema_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R crosses above -50 (mean reversion) OR loss of volume/spike OR trend reversal
            if wr > -50 or not volume_spike[i] or curr_close < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -50 (mean reversion) OR loss of volume/spike OR trend reversal
            if wr < -50 or not volume_spike[i] or curr_close > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0