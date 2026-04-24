#!/usr/bin/env python3
"""
Hypothesis: Daily Donchian(20) breakout with weekly EMA(34) trend filter and daily volume spike confirmation.
- Primary timeframe: 1d for entries/exits.
- HTF: 1w EMA(34) for trend direction (bullish if price > EMA34, bearish if price < EMA34).
- Volume: Current daily volume > 2.0 * 20-period daily volume MA to avoid false breakouts.
- Entry: Long when price breaks above Donchian(20) high AND 1w EMA34 trend bullish AND volume spike.
         Short when price breaks below Donchian(20) low AND 1w EMA34 trend bearish AND volume spike.
- Exit: Opposite Donchian breakout or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
- Works in both bull and bear: Trend filter prevents counter-trend trades, volume spike confirms momentum.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period) on daily
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get weekly data for EMA(34) trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate EMA(34) on weekly close
    ema_34 = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period volume MA on daily
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to daily
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # Volume confirmation: current daily volume > 2.0 * 20-period daily volume MA
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need enough weekly bars for EMA34 and daily bars for Donchian/volume
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_34_val = ema_34_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        upper_donchian = donchian_high[i]
        lower_donchian = donchian_low[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish breakout: price breaks above upper Donchian AND 1w EMA34 bullish (price > EMA34)
                if curr_high > upper_donchian and ema_34_val > 0 and curr_close > ema_34_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below lower Donchian AND 1w EMA34 bearish (price < EMA34)
                elif curr_low < lower_donchian and ema_34_val > 0 and curr_close < ema_34_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below lower Donchian OR loss of volume confirmation
            if curr_low < lower_donchian or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above upper Donchian OR loss of volume confirmation
            if curr_high > upper_donchian or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA34Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0