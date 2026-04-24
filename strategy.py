#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 1d for entries/exits.
- HTF: 1w EMA34 for trend direction (bullish if price > EMA34, bearish if price < EMA34).
- Volume: Current 1d volume > 2.0 * 20-period volume MA to avoid false breakouts.
- Entry: Long when price breaks above Donchian(20) high AND 1w EMA34 bullish AND volume spike.
         Short when price breaks below Donchian(20) low AND 1w EMA34 bearish AND volume spike.
- Exit: Opposite Donchian breakout or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
Donchian breakouts capture strong trends, EMA34 filter avoids counter-trend trades, volume confirmation reduces false signals.
Works in both bull and bear markets by only taking trades in the direction of the 1w trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian(20) channels
    period20_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34
    df_1w_close = df_1w['close'].values
    ema_34_1w = pd.Series(df_1w_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period volume MA on 1w
    df_1w_volume = df_1w['volume'].values
    vol_ma_20_1w = pd.Series(df_1w_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 1d
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    vol_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20_1w)
    
    # Volume confirmation: current 1d volume > 2.0 * 20-period 1w volume MA (aligned)
    volume_spike = volume > (2.0 * vol_ma_20_1w_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34, 20)  # Need enough bars for Donchian and 1w indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(period20_high[i]) or np.isnan(period20_low[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        upper_donchian = period20_high[i]
        lower_donchian = period20_low[i]
        ema_val = ema_34_1w_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish: price breaks above Donchian high AND 1w EMA34 bullish (price > EMA)
                if curr_high > upper_donchian and curr_close > ema_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish: price breaks below Donchian low AND 1w EMA34 bearish (price < EMA)
                elif curr_low < lower_donchian and curr_close < ema_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian low OR loss of volume confirmation
            if curr_low < lower_donchian or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high OR loss of volume confirmation
            if curr_high > upper_donchian or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA34_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0