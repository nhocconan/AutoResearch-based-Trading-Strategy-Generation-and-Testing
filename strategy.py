#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R reversal with 1d EMA34 trend filter and volume confirmation.
Long when Williams %R crosses above -80 from below AND 1d EMA34 is rising AND volume > 1.3x 20-period average.
Short when Williams %R crosses below -20 from above AND 1d EMA34 is falling AND volume > 1.3x 20-period average.
Exit when Williams %R reaches opposite extreme (-20 for long, -80 for short) or EMA34 reverses direction.
Williams %R is effective in ranging markets which dominate 2025 BTC/ETH action, while EMA34 filter avoids whipsaws.
Target: 75-200 total trades over 4 years (19-50/year).
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
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R (14-period)
    lookback = 14
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    williams_r = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        hh = highest_high[i]
        ll = lowest_low[i]
        if hh != ll:  # avoid division by zero
            williams_r[i] = (hh - close[i]) / (hh - ll) * -100
        else:
            williams_r[i] = -50  # midpoint when range is zero
    
    # 20-period volume average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback - 1, 34, 20)  # Williams %R (14), EMA34 (34), volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_34_aligned[i]
        wr = williams_r[i]
        vol_ma_val = vol_ma[i]
        
        # Calculate Williams %R crossover signals
        wr_long_signal = False
        wr_short_signal = False
        if i >= start_idx + 1:
            wr_prev = williams_r[i-1]
            # Long: crosses above -80 from below
            if wr_prev <= -80 and wr > -80:
                wr_long_signal = True
            # Short: crosses below -20 from above
            if wr_prev >= -20 and wr < -20:
                wr_short_signal = True
        
        # Calculate EMA34 slope for trend direction
        ema_rising = False
        ema_falling = False
        if i >= start_idx + 1:
            ema_prev = ema_34_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        
        if position == 0:
            # Long: Williams %R crosses above -80 AND EMA34 rising AND volume confirmation
            if wr_long_signal and ema_rising and volume[i] > 1.3 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 AND EMA34 falling AND volume confirmation
            elif wr_short_signal and ema_falling and volume[i] > 1.3 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Williams %R reaches -20 (overbought) OR EMA34 starts falling
                if wr >= -20 or (i >= start_idx + 1 and ema_val < ema_34_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: Williams %R reaches -80 (oversold) OR EMA34 starts rising
                if wr <= -80 or (i >= start_idx + 1 and ema_val > ema_34_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsR_Reversal_1dEMA34_Trend_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0