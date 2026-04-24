#!/usr/bin/env python3
"""
Hypothesis: 1d Williams %R with 1-week EMA34 trend filter and volume confirmation.
- Williams %R (14) identifies overbought/oversold conditions: < -80 oversold, > -20 overbought.
- 1-week EMA34 provides higher-timeframe trend filter to align with dominant momentum.
- Volume confirmation (>1.5x 20-period average) reduces false signals.
- Long: Williams %R crosses above -80 from below + price > 1w EMA34 + volume confirmation.
- Short: Williams %R crosses below -20 from above + price < 1w EMA34 + volume confirmation.
- Exit: Opposite Williams %R crossover or volume divergence.
- Position size 0.25 balances profit and drawdown control.
- Target trades: 30-100 total over 4 years (7-25/year) to minimize fee drag.
- Works in bull/bear markets via 1-week trend filter and mean-reversion logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w data ONCE before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # 1d Williams %R (14)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # 1w EMA34 trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(williams_r[i-1]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams %R crossovers
        wr_cross_above_80 = williams_r[i-1] <= -80 and williams_r[i] > -80  # Oversold to normal
        wr_cross_below_20 = williams_r[i-1] >= -20 and williams_r[i] < -20  # Overbought to normal
        
        # Volume confirmation
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Only trade with volume confirmation
            if volume_confirm:
                # Long: Williams %R crosses above -80 + price > 1w EMA34 (bullish higher-timeframe trend)
                if wr_cross_above_80 and close[i] > ema_34_1w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R crosses below -20 + price < 1w EMA34 (bearish higher-timeframe trend)
                elif wr_cross_below_20 and close[i] < ema_34_1w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -20 (overbought) OR volume divergence
            if wr_cross_below_20 or (volume[i] < 0.5 * vol_ma[i] and williams_r[i] > -50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -80 (oversold) OR volume divergence
            if wr_cross_above_80 or (volume[i] < 0.5 * vol_ma[i] and williams_r[i] < -50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsR_1wEMA34_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0