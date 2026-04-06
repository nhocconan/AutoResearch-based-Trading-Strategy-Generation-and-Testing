#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1-day trend filter and volume confirmation.
# Williams %R (14-period) identifies overbought/oversold conditions for mean reversion.
# 1-day EMA (20-period) filters trades to align with daily trend direction.
# Volume confirmation (current volume > 1.5x 20-period average) ensures breakout strength.
# Works in bull markets via oversold bounces and in bear markets via overbought reversals.
# Target: 75-200 total trades over 4 years (19-50/year).

name = "12h_williamsr_1d_trend_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period)
    willr = np.full(n, np.nan)
    for i in range(13, n):
        highest_high = np.max(high[i-13:i+1])
        lowest_low = np.min(low[i-13:i+1])
        if highest_high - lowest_low != 0:
            willr[i] = (highest_high - close[i]) / (highest_high - lowest_low) * -100
        else:
            willr[i] = -50  # neutral when no range
    
    # 1-day trend filter: 20-period EMA on daily closes
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_20d = np.full(len(close_1d), np.nan)
    for i in range(len(close_1d)):
        if i < 19:
            ema_20d[i] = np.nan
        elif i == 19:
            ema_20d[i] = np.mean(close_1d[0:20])
        else:
            ema_20d[i] = close_1d[i] * 2/(20+1) + ema_20d[i-1] * (1 - 2/(20+1))
    ema_20d_aligned = align_htf_to_ltf(prices, df_1d, ema_20d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(willr[i]) or np.isnan(ema_20d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: Williams %R exits overbought or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.5 * atr_approx
            
            if (willr[i] > -20 or  # exited overbought
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Williams %R exits oversold or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.5 * atr_approx
            
            if (willr[i] < -80 or  # exited oversold
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and daily trend filter
            if volume_filter:
                # Oversold bounce with daily uptrend
                if (willr[i] < -80 and willr[i-1] >= -80 and 
                    close[i] > ema_20d_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Overbought reversal with daily downtrend
                elif (willr[i] > -20 and willr[i-1] <= -20 and 
                      close[i] < ema_20d_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals