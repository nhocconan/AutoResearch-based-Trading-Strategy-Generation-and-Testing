#!/usr/bin/env python3
"""
Hypothesis: 1h Donchian channel breakout with 4h volume spike and 1d EMA trend filter.
- Primary timeframe: 1h for execution, HTF: 4h for volume confirmation, 1d for EMA trend.
- EMA50 > EMA200 on 1d indicates bullish trend (long bias), EMA50 < EMA200 indicates bearish trend (short bias).
- Entry: Long when price breaks above Donchian(20) high AND 4h volume > 1.5 * 20-period MA AND EMA50 > EMA200.
         Short when price breaks below Donchian(20) low AND 4h volume > 1.5 * 20-period MA AND EMA50 < EMA200.
- Exit: Opposite Donchian breakout or EMA trend flip.
- Volume confirmation avoids false breakouts in low-volatility periods.
- Discrete signal size: 0.20 to limit drawdown and reduce fee churn.
- Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
- Session filter: 08-20 UTC to avoid low-volume Asian session noise.
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
    
    # Get 4h data for volume confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h volume MA (20-period)
    volume_4h = df_4h['volume'].values
    volume_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_spike_4h = volume_4h > (1.5 * volume_ma_4h)
    volume_spike_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_spike_4h)
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 and EMA200
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_bullish = ema50_1d > ema200_1d  # Bullish trend when EMA50 > EMA200
    ema_bearish = ema50_1d < ema200_1d  # Bearish trend when EMA50 < EMA200
    ema_bullish_aligned = align_htf_to_ltf(prices, df_1d, ema_bullish)
    ema_bearish_aligned = align_htf_to_ltf(prices, df_1d, ema_bearish)
    
    # Calculate 1h Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Session filter: 08-20 UTC (avoid low-volume Asian session)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20, 50, 20)  # 4h vol MA, Donchian, 1d EMA200
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if data not ready
        if (np.isnan(volume_spike_4h_aligned[i]) or np.isnan(ema_bullish_aligned[i]) or 
            np.isnan(ema_bearish_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        vol_spike = volume_spike_4h_aligned[i]
        ema_bull = ema_bullish_aligned[i]
        ema_bear = ema_bearish_aligned[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        
        if position == 0:
            # Check for entry signals
            if vol_spike:
                # Bullish breakout in bullish trend
                if ema_bull and curr_high > upper:
                    signals[i] = 0.20
                    position = 1
                # Bearish breakout in bearish trend
                elif ema_bear and curr_low < lower:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian low OR trend turns bearish
            if curr_low < lower or ema_bear:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above Donchian high OR trend turns bullish
            if curr_high > upper or ema_bull:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Donchian20_4hVolumeSpike_1dEMATrend_v1"
timeframe = "1h"
leverage = 1.0