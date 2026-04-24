#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R mean reversion with 1w trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w EMA50 for trend filter (price > EMA50 = uptrend, price < EMA50 = downtrend).
- Williams %R(14): Long when %R crosses above -80 from below (oversold bounce) in uptrend;
                    Short when %R crosses below -20 from above (overbought rejection) in downtrend.
- Volume confirmation: volume > 1.5 * 6h volume MA(50) to ensure conviction.
- Exit: Opposite %R signal (long exits when %R < -50, short exits when %R > -50) for mean reversion.
- Signal size: 0.25 discrete to control fee drag.
- Williams %R captures exhaustion moves in both bull (bounce from oversold) and bear (rejection from overbought) markets,
  while 1w EMA50 ensures we trade with the major trend. Volume filter avoids low-conviction whipsaws.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 55:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams %R(14) for 6h timeframe
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    hh_ll = highest_high - lowest_low
    hh_ll = np.where(hh_ll == 0, 1e-10, hh_ll)
    williams_r = (highest_high - close) / hh_ll * -100
    
    # Calculate volume MA(50) for 6h timeframe
    vol_ma_6h = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 50, 14)  # EMA50 needs 50, volume MA needs 50, %R needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_wr = williams_r[i]
        curr_volume = volume[i]
        prev_wr = williams_r[i-1]
        
        # Volume confirmation: 1.5x threshold
        vol_confirm = curr_volume > 1.5 * vol_ma_6h[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Williams %R crosses above -80 from below (oversold bounce) AND price > 1w EMA50 (uptrend)
                if curr_wr > -80 and prev_wr <= -80 and curr_close > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R crosses below -20 from above (overbought rejection) AND price < 1w EMA50 (downtrend)
                elif curr_wr < -20 and prev_wr >= -20 and curr_close < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit when %R falls below -50 (mean reversion)
            if curr_wr < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when %R rises above -50 (mean reversion)
            if curr_wr > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_MeanReversion_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0