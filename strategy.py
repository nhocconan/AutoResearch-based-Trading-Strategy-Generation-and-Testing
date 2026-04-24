#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) Breakout with 1d EMA34 Trend Filter and Volume Spike.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d EMA34 for trend filter (price > EMA34 = uptrend, price < EMA34 = downtrend).
- Entry: Long when price breaks above Donchian(20) high AND price > 1d EMA34 AND volume > 2.0 * 4h volume MA(20);
         Short when price breaks below Donchian(20) low AND price < 1d EMA34 AND volume > 2.0 * 4h volume MA(20).
- Exit: Long exits when price crosses below Donchian(20) low; Short exits when price crosses above Donchian(20) high.
- Signal size: 0.30 discrete to balance capture and fee control.
- Works in bull (buying breakouts in uptrend) and bear (selling breakdowns in downtrend) with reduced whipsaws.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA34 to 4h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate Donchian(20) on 4h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 4h data for volume MA(20)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 20)  # EMA34 needs 34, Donchian needs 20, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Volume confirmation: 2.0x threshold
        vol_confirm = curr_volume > 2.0 * vol_ma[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: price breaks above Donchian high AND price > 1d EMA34 (uptrend)
                if curr_close > donchian_high[i] and curr_close > ema_34_aligned[i]:
                    signals[i] = 0.30
                    position = 1
                # Short: price breaks below Donchian low AND price < 1d EMA34 (downtrend)
                elif curr_close < donchian_low[i] and curr_close < ema_34_aligned[i]:
                    signals[i] = -0.30
                    position = -1
        elif position == 1:
            # Long position: exit when price crosses below Donchian low
            if curr_close < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short position: exit when price crosses above Donchian high
            if curr_close > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0