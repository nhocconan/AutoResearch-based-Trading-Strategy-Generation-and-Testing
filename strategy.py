#!/usr/bin/env python3
"""
Hypothesis: 12h strategy using 1d Williams %R extremes with 1w EMA34 trend filter and volume confirmation.
Long when Williams %R(14) crosses above -80 (oversold) AND price > 1w EMA34 AND volume > 1.3x 20-period average.
Short when Williams %R(14) crosses below -20 (overbought) AND price < 1w EMA34 AND volume > 1.3x 20-period average.
Exit when Williams %R returns to -50 (mean reversion) or ATR trailing stop hit (2.5*ATR from extreme).
Uses discrete position sizing (0.25) to control drawdown and fee churn.
Designed for 12h timeframe to target 12-37 trades/year per symbol (50-150 total over 4 years).
Williams %R is a momentum oscillator that works well in ranging markets; EMA34 filter ensures we trade with the weekly trend.
Volume confirmation avoids false breakouts in low-liquidity periods.
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
    
    # Calculate 1d Williams %R (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    highest_high = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - df_1d['close'].values) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align 1d Williams %R to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    ema_34 = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # Volume average (20-period) on 12h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    extreme_since_entry = 0.0  # highest for long, lowest for short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 34, 20)  # Williams needs 14, EMA needs 34, vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        wr_val = williams_r_aligned[i]
        ema_34_val = ema_34_aligned[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 (from below) AND price > 1w EMA34 AND volume spike
            wr_prev = williams_r_aligned[i-1]
            if (wr_val > -80 and wr_prev <= -80 and price > ema_34_val and volume[i] > 1.3 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
                extreme_since_entry = price  # track highest for long
            # Short: Williams %R crosses below -20 (from above) AND price < 1w EMA34 AND volume spike
            elif (wr_val < -20 and wr_prev >= -20 and price < ema_34_val and volume[i] > 1.3 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
                extreme_since_entry = price  # track lowest for short
        else:
            # Update extreme since entry for trailing stop
            if position == 1:
                extreme_since_entry = max(extreme_since_entry, price)
            elif position == -1:
                extreme_since_entry = min(extreme_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Williams %R returns to -50 (mean reversion)
            if position == 1 and wr_val >= -50:
                exit_signal = True
            elif position == -1 and wr_val <= -50:
                exit_signal = True
            
            # ATR-based trailing stop: 2.5 * ATR from extreme since entry
            if position == 1 and price < extreme_since_entry - 2.5 * atr_val:
                exit_signal = True
            elif position == -1 and price > extreme_since_entry + 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                extreme_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsR_Extremes_1wEMA34_Trend_VolumeConfirmation_ATRTrailingStop"
timeframe = "12h"
leverage = 1.0