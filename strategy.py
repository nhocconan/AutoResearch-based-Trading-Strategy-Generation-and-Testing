#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R mean reversion with 1d EMA trend filter and volume spike confirmation.
Long when Williams %R < -80 (oversold) AND price > 1d EMA50 (uptrend bias) AND volume > 2x 20-period average.
Short when Williams %R > -20 (overbought) AND price < 1d EMA50 (downtrend bias) AND volume > 2x 20-period average.
Exit when Williams %R crosses above -50 for longs or below -50 for shorts.
Uses 1d HTF EMA for trend alignment to avoid counter-trend trades. Target: 50-150 total trades over 4 years (12-37/year).
Williams %R identifies extreme reversals; EMA filter ensures we trade with the higher timeframe trend; volume spike confirms conviction.
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
    
    # Calculate 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 6h Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 20)  # williams_r (14), vol_ma (20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        wr = williams_r[i]
        ema_trend = ema_1d_aligned[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: Oversold (WR < -80) AND uptrend (price > EMA) AND volume spike
            if wr < -80 and price > ema_trend and volume[i] > 2.0 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Overbought (WR > -20) AND downtrend (price < EMA) AND volume spike
            elif wr > -20 and price < ema_trend and volume[i] > 2.0 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit when WR crosses -50 (mean reversion complete)
            if position == 1 and wr > -50:  # Long exit
                exit_signal = True
            elif position == -1 and wr < -50:  # Short exit
                exit_signal = True
            else:
                exit_signal = False
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_MeanReversion_1dEMA50_Trend_VolumeSpike_WR50Exit"
timeframe = "6h"
leverage = 1.0