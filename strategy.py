#!/usr/bin/env python3
"""
1d_donchian_20_breakout_1w_trend_volume_v1
Hypothesis: On 1d timeframe, enter long when price breaks above 20-day Donchian high with weekly uptrend and volume confirmation; enter short when price breaks below 20-day Donchian low with weekly downtrend and volume confirmation. Exit on opposite signal or when price returns to 20-day Donchian middle. Weekly trend filter avoids counter-trend trades in choppy markets. Targets 10-25 trades/year to minimize fee drag and improve generalization.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_20_breakout_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ATR for volatility filter and position sizing
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation (20-day average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get weekly data for trend filter (calculate once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 21-period EMA for weekly trend
    close_1w = df_1w['close'].values
    ema_21 = pd.Series(close_1w).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # Align weekly EMA to daily (shifted by 1 week to avoid look-ahead)
    ema_21_aligned = align_htf_to_ltf(prices, df_1w, ema_21)
    
    # Calculate 20-day Donchian channels (using daily data)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_middle = (high_20 + low_20) / 2
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or atr[i] <= 0 or 
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0 or
            np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(ema_21_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit on short signal (price breaks below Donchian low with volume)
            if close[i] < low_20[i] and vol_confirm:
                exit_long = True
            # Exit when price returns to Donchian middle (mean reversion)
            elif abs(close[i] - donchian_middle[i]) < 0.5 * atr[i]:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit on long signal (price breaks above Donchian high with volume)
            if close[i] > high_20[i] and vol_confirm:
                exit_short = True
            # Exit when price returns to Donchian middle (mean reversion)
            elif abs(close[i] - donchian_middle[i]) < 0.5 * atr[i]:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high with volume confirmation and weekly uptrend
            long_entry = (close[i] > high_20[i] and 
                         vol_confirm and 
                         close[i] > ema_21_aligned[i])
            
            # Short entry: price breaks below Donchian low with volume confirmation and weekly downtrend
            short_entry = (close[i] < low_20[i] and 
                          vol_confirm and 
                          close[i] < ema_21_aligned[i])
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals