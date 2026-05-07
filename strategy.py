#!/usr/bin/env python3
# 12h_KeltnerBreakout_1dTrend_Volume
# Hypothesis: Combines Keltner Channel breakout with 1-day EMA trend filter and volume confirmation.
# Keltner Channel uses ATR to set dynamic bands, adapting to volatility regimes.
# Trades breakouts in the direction of the 1-day trend, avoiding counter-trend moves.
# Volume spike confirms breakout strength. Designed for 12h timeframe to limit trade frequency
# and reduce fee impact, targeting 12-30 trades/year per symbol.

name = "12h_KeltnerBreakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1-day EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_12h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(20) for Keltner Channel
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channel: 20-period EMA +/- 2 * ATR
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper_keltner = ema_20 + 2 * atr
    lower_keltner = ema_20 - 2 * atr
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema_34_1d_12h[i]) or np.isnan(ema_20[i]) or 
            np.isnan(atr[i]) or np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.5 * 20-period average volume
        vol_ma_20 = np.mean(volume[max(0, i-19):i+1]) if i >= 19 else np.nan
        volume_spike = not np.isnan(vol_ma_20) and volume[i] > 1.5 * vol_ma_20
        
        if position == 0:
            # Long: Close above upper Keltner + above 1d EMA34 + volume spike
            if close[i] > upper_keltner[i] and close[i] > ema_34_1d_12h[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Close below lower Keltner + below 1d EMA34 + volume spike
            elif close[i] < lower_keltner[i] and close[i] < ema_34_1d_12h[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below EMA20 (middle of Keltner) or below 1d EMA34
            if close[i] < ema_20[i] or close[i] < ema_34_1d_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above EMA20 or above 1d EMA34
            if close[i] > ema_20[i] or close[i] > ema_34_1d_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals