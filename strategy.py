#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R reversal with 1d EMA(34) trend filter and volume confirmation.
- Primary timeframe: 6h for entries/exits.
- HTF: 1d EMA(34) for trend direction (bullish if price > EMA34, bearish if price < EMA34).
- Williams %R(14) on 6h for mean reversion signals: long when %R crosses above -80 from below,
  short when %R crosses below -20 from above.
- Volume confirmation: current 6h volume > 1.8 * 20-period volume MA to filter low-momentum noise.
- Exit: Opposite Williams %R cross (%R crosses below -50 for long, above -50 for short) or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Why it works: Williams %R captures short-term exhaustion in trends; EMA34 filter ensures alignment with daily momentum;
  volume confirmation avoids false signals in low-liquidity periods. Effective in both bull (buy dips) and bear (sell rallies).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R (14-period) on 6h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).values
    
    # Get 1d data for EMA(34) trend and volume MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d close
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period volume MA on 1d
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 6h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Volume confirmation: current 6h volume > 1.8 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (1.8 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 14, 20)  # Need enough 1d bars for EMA34 and volume MA, and 6h bars for Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_34_val = ema_34_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        williams_r_val = williams_r[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish reversal: Williams %R crosses above -80 from below AND 1d EMA34 bullish (price > EMA34)
                if i > start_idx and williams_r[i-1] <= -80 and williams_r_val > -80 and curr_close > ema_34_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish reversal: Williams %R crosses below -20 from above AND 1d EMA34 bearish (price < EMA34)
                elif i > start_idx and williams_r[i-1] >= -20 and williams_r_val < -20 and curr_close < ema_34_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -50 OR loss of volume confirmation
            if i > start_idx and williams_r[i-1] >= -50 and williams_r_val < -50:
                signals[i] = 0.0
                position = 0
            elif not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -50 OR loss of volume confirmation
            if i > start_idx and williams_r[i-1] <= -50 and williams_r_val > -50:
                signals[i] = 0.0
                position = 0
            elif not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_1dEMA34Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0