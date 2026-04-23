#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R mean reversion with 1d EMA34 trend filter and volume confirmation.
Long when Williams %R < -80 (oversold) AND 1d EMA34 is rising AND volume > 1.3x 20-period average.
Short when Williams %R > -20 (overbought) AND 1d EMA34 is falling AND volume > 1.3x 20-period average.
Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts) OR EMA34 reverses.
Uses 1d HTF for EMA34 trend to avoid whipsaws in ranging markets. Target: 75-200 total trades over 4 years (19-50/year).
Williams %R is effective at catching reversals in both bull and bear markets, especially when combined with trend and volume filters.
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
    
    # Calculate 14-period Williams %R
    lookback = 14
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    williams_r = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i]) * -100
        else:
            williams_r[i] = -50  # neutral when range is zero
    
    # 20-period volume average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback - 1, 34, 20)  # Williams %R (14), EMA34 (34), volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(williams_r[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_34_aligned[i]
        wr = williams_r[i]
        vol_ma_val = vol_ma[i]
        
        # Calculate EMA34 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_34_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        if position == 0:
            # Long: Oversold (WR < -80) AND EMA34 rising AND volume confirmation
            if wr < -80 and ema_rising and volume[i] > 1.3 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Overbought (WR > -20) AND EMA34 falling AND volume confirmation
            elif wr > -20 and ema_falling and volume[i] > 1.3 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: WR crosses above -50 OR EMA34 starts falling
                if wr > -50 or (i >= start_idx + 1 and ema_val < ema_34_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: WR crosses below -50 OR EMA34 starts rising
                if wr < -50 or (i >= start_idx + 1 and ema_val > ema_34_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsR_MeanReversion_1dEMA34_Trend_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0