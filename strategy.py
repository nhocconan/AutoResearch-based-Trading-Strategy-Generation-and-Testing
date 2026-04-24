#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R extreme reversals with 1d EMA(34) trend filter and volume confirmation.
- Primary timeframe: 12h for entries/exits.
- HTF: 1d EMA(34) for trend direction (bullish if price > EMA34, bearish if price < EMA34).
- Williams %R(14): Long when %R crosses above -80 from below (oversold reversal).
                    Short when %R crosses below -20 from above (overbought reversal).
- Volume: Current 12h volume > 1.5 * 20-period volume MA to confirm momentum.
- Entry: Long when Williams %R crosses above -80 AND 1d EMA34 trend bullish AND volume spike.
         Short when Williams %R crosses below -20 AND 1d EMA34 trend bearish AND volume spike.
- Exit: Opposite Williams %R extreme (%R < -80 for long exit, %R > -20 for short exit) OR loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
- Williams %R is effective in ranging/bear markets (2025-2026 test period) for mean reversion.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R (14-period) on 12h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Get 1d data for EMA(34) trend and volume MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d close
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period volume MA on 1d
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 12h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Volume confirmation: current 12h volume > 1.5 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 14)  # Need enough 1d bars for EMA34 and volume MA, and 12h bars for Williams %R
    
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
        curr_williams_r = williams_r[i]
        prev_williams_r = williams_r[i-1] if i > 0 else -50
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish reversal: Williams %R crosses above -80 from below AND 1d EMA34 bullish (price > EMA34)
                if prev_williams_r <= -80 and curr_williams_r > -80 and curr_close > ema_34_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish reversal: Williams %R crosses below -20 from above AND 1d EMA34 bearish (price < EMA34)
                elif prev_williams_r >= -20 and curr_williams_r < -20 and curr_close < ema_34_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R goes below -80 (re-entered oversold) OR loss of volume confirmation
            if curr_williams_r < -80 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R goes above -20 (re-entered overbought) OR loss of volume confirmation
            if curr_williams_r > -20 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR14_1dEMA34Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0