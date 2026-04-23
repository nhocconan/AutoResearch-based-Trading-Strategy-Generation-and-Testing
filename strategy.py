#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R mean reversion with 1d trend filter and volume confirmation.
Long when Williams %R < -80 (oversold) AND price > 1d EMA34 (uptrend) AND volume > 1.5x 20-period average.
Short when Williams %R > -20 (overbought) AND price < 1d EMA34 (downtrend) AND volume > 1.5x 20-period average.
Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts).
Uses discrete position sizing (0.25) to target 12-37 trades/year on 6h timeframe.
Williams %R is effective in ranging markets which dominate BTC/ETH 2025+ bearish regime.
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
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Williams %R(14) on 6h timeframe
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Avoid division by zero
    rr = highest_high - lowest_low
    rr[rr == 0] = 1e-10
    williams_r = -100 * ((highest_high - close) / rr)
    
    # Volume average (20-period for 6h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 14, 20)  # EMA34 needs 34, WilliamsR needs 14, vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        ema34_val = ema34_1d_aligned[i]
        wr = williams_r[i]
        
        if position == 0:
            # Long: Oversold AND uptrend AND volume spike
            if wr < -80 and price > ema34_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Overbought AND downtrend AND volume spike
            elif wr > -20 and price < ema34_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Williams %R crosses above -50 (for longs) or below -50 (for shorts)
            if position == 1 and wr > -50:
                exit_signal = True
            elif position == -1 and wr < -50:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_14_1dEMA34_Trend_VolumeConfirmation_ExitAt_Minus50"
timeframe = "6h"
leverage = 1.0