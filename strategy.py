#!/usr/bin/env python3
# Hypothesis: 12h Bollinger Band breakout with 1d EMA34 trend filter and volume confirmation.
# Long when close breaks above upper Bollinger Band (20,2) AND price > 1d EMA34 AND volume > 1.5x 20-period average volume.
# Short when close breaks below lower Bollinger Band (20,2) AND price < 1d EMA34 AND volume > 1.5x 20-period average volume.
# Exit when price crosses back inside Bollinger Bands OR trend filter reverses.
# Uses discrete position sizing (0.25) to limit fee churn and manage drawdown.
# Designed for low trade frequency (~12-37/year) by requiring confluence of Bollinger breakout, daily trend, and volume spike.
# Bollinger Bands capture volatility expansion; EMA34 filters for higher timeframe trend; volume confirms conviction.
# Effective in both bull and bear markets by trading breakouts with trend and volume confirmation.

name = "12h_Bollinger_Breakout_1dEMA34_Volume_v1"
timeframe = "12h"
leverage = 1.0

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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(34) on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Bollinger Bands (20,2) on 12h close
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20
    
    # Volume confirmation: volume > 1.5x 20-period average volume
    avg_vol20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > 1.5 * avg_vol20
    
    # Track entry price for carry-forward
    entry_price = np.full(n, np.nan)
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after sufficient data for Bollinger Bands
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(sma20[i]) or np.isnan(std20[i]) or \
           np.isnan(avg_vol20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: close breaks above upper BB, price > 1d EMA34, volume confirmation
            if close[i] > upper_bb[i] and close[i] > ema34_1d_aligned[i] and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
                entry_price[i] = close[i]
            # SHORT: close breaks below lower BB, price < 1d EMA34, volume confirmation
            elif close[i] < lower_bb[i] and close[i] < ema34_1d_aligned[i] and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
                entry_price[i] = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses back inside Bollinger Bands OR trend filter reverses (price < 1d EMA34)
            if close[i] < upper_bb[i] and close[i] > lower_bb[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.25
                entry_price[i] = entry_price[i-1]
        elif position == -1:
            # EXIT SHORT: price crosses back inside Bollinger Bands OR trend filter reverses (price > 1d EMA34)
            if close[i] < upper_bb[i] and close[i] > lower_bb[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = -0.25
                entry_price[i] = entry_price[i-1]
    
    return signals