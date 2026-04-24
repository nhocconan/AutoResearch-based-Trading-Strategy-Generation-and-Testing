#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h volume spike and 12h EMA(50) trend filter.
- Primary: 4h timeframe for entries/exits.
- HTF: 12h EMA(50) for trend direction (bullish if price > EMA50, bearish if price < EMA50).
- Volume: Current 4h volume > 2.0 * 20-period 12h volume MA (aligned) to confirm breakout strength.
- Entry: Long when price breaks above Donchian(20) high AND 12h EMA50 trend bullish AND volume spike.
         Short when price breaks below Donchian(20) low AND 12h EMA50 trend bearish AND volume spike.
- Exit: Opposite Donchian breakout or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
- Works in bull via breakout momentum and in bear via short-side breakouts with volume confirmation.
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
    
    # Calculate Donchian channels (20-period) on 4h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 12h data for EMA(50) trend and volume MA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h close
    ema_50 = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 20-period volume MA on 12h
    vol_ma_12h = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 4h
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Volume confirmation: current 4h volume > 2.0 * 20-period 12h volume MA (aligned)
    volume_spike = volume > (2.0 * vol_ma_12h_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need enough 12h bars for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_50_val = ema_50_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        upper_donchian = donchian_high[i]
        lower_donchian = donchian_low[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish breakout: price breaks above upper Donchian AND 12h EMA50 bullish (price > EMA50)
                if curr_high > upper_donchian and ema_50_val > 0 and curr_close > ema_50_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below lower Donchian AND 12h EMA50 bearish (price < EMA50)
                elif curr_low < lower_donchian and ema_50_val > 0 and curr_close < ema_50_val:
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

name = "4h_Donchian20_12hEMA50Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0