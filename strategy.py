#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R mean reversion with 1d EMA34 trend filter and volume confirmation.
Long when Williams %R < -80 (oversold) AND price > 1d EMA34 (uptrend) AND volume > 1.5x 20-period MA.
Short when Williams %R > -20 (overbought) AND price < 1d EMA34 (downtrend) AND volume > 1.5x 20-period MA.
Exit when Williams %R returns to -50 (mean reversion) or opposite extreme is hit.
Designed for ~15-25 trades/year with mean reversion edge in ranging markets and trend filter to avoid false signals in strong trends.
Williams %R identifies exhaustion points; 1d EMA34 ensures higher timeframe alignment; volume confirmation reduces false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 14)  # need EMA34, volume MA20, Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: close > 1d EMA34 = uptrend, close < 1d EMA34 = downtrend
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        # Volume filter: 6h volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # Williams %R conditions
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        mean_revert = abs(williams_r[i] + 50) < 5  # Near -50 (mean reversion)
        
        if position == 0:
            # Long: Oversold AND uptrend AND volume confirmation
            if oversold and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Overbought AND downtrend AND volume confirmation
            elif overbought and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: mean reversion or opposite extreme
            exit_signal = False
            if position == 1:
                exit_signal = mean_revert or overbought
            elif position == -1:
                exit_signal = mean_revert or oversold
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_MeanReversion_1dEMA34_Trend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0