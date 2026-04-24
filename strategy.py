#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
- Primary timeframe: 12h for entries/exits.
- HTF: 1d EMA34 for trend direction (bullish if close > EMA34, bearish if close < EMA34).
- Volume: Current 12h volume > 1.8 * 20-period volume MA to confirm breakout strength.
- Entry: Long when price breaks above Donchian(20) high AND 1d EMA34 bullish AND volume spike.
         Short when price breaks below Donchian(20) low AND 1d EMA34 bearish AND volume spike.
- Exit: Opposite Donchian breakout or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
Donchian breakouts capture strong momentum moves, EMA34 filters counter-trend noise,
and volume confirmation avoids false breakouts in low-liquidity periods.
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
    
    # Calculate 20-period volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    df_1d_close = df_1d['close'].values
    ema_34_1d = pd.Series(df_1d_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # 1d EMA34 trend: 1 if bullish (close > EMA34), -1 if bearish (close < EMA34), 0 otherwise
    ema34_trend = np.where(df_1d_close > ema_34_1d, 1, np.where(df_1d_close < ema_34_1d, -1, 0))
    
    # Align HTF indicators to 12h
    ema34_trend_aligned = align_htf_to_ltf(prices, df_1d, ema34_trend)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34)  # Need enough bars for Donchian and 1d EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(period20_high[i]) or np.isnan(period20_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema34_trend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume[i] > (1.8 * vol_ma[i])
        trend = ema34_trend_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if vol_spike:
                # Bullish: price breaks above Donchian high AND 1d EMA34 bullish
                if curr_high > period20_high[i] and trend == 1:
                    signals[i] = 0.25
                    position = 1
                # Bearish: price breaks below Donchian low AND 1d EMA34 bearish
                elif curr_low < period20_low[i] and trend == -1:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian low OR loss of volume confirmation
            if curr_low < period20_low[i] or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high OR loss of volume confirmation
            if curr_high > period20_high[i] or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0