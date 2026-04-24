#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
- Primary timeframe: 12h for entries/exits.
- HTF: 1d EMA34 for trend direction (bullish if close > EMA34, bearish if close < EMA34).
- Volume: Current 12h volume > 1.5 * 20-period volume MA to avoid low-momentum breakouts.
- Entry: Long when price breaks above Donchian(20) high AND 1d EMA34 bullish AND volume spike.
         Short when price breaks below Donchian(20) low AND 1d EMA34 bearish AND volume spike.
- Exit: Opposite Donchian breakout (short exit on upper band, long exit on lower band).
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
Donchian breakouts capture strong momentum moves; EMA34 filter ensures trend alignment.
Volume confirmation avoids false breakouts in low-liquidity periods.
Works in bull markets via breakouts and in bear markets via short breakdowns.
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
    
    # Calculate Donchian channels on 12h
    period = 20
    donchian_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    donchian_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Get 1d data for EMA34 and volume MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period volume MA on 1d
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 12h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Determine 1d trend: 1 if bullish (close > EMA34), -1 if bearish (close < EMA34), 0 otherwise
    # Use aligned close prices for comparison
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    trend_1d = np.where(close_1d_aligned > ema_34_aligned, 1, np.where(close_1d_aligned < ema_34_aligned, -1, 0))
    
    # Volume confirmation: current 12h volume > 1.5 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (1.5 * vol_ma_20_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(period, 34, 20)  # Need enough bars for Donchian and 1d indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(trend_1d[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for entry signals with volume spike and trend alignment
            if volume_spike[i]:
                # Bullish breakout: price above Donchian high AND 1d trend bullish
                if high[i] > donchian_high[i] and trend_1d[i] == 1:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakdown: price below Donchian low AND 1d trend bearish
                elif low[i] < donchian_low[i] and trend_1d[i] == -1:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian low
            if low[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high
            if high[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0