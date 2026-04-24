#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w EMA50 for trend filter (price > EMA50 = uptrend, price < EMA50 = downtrend).
- Entry: Long when price breaks above Donchian(20) upper band AND price > 1w EMA50 AND volume > 2.0 * 1d volume MA(20);
         Short when price breaks below Donchian(20) lower band AND price < 1w EMA50 AND volume > 2.0 * 1d volume MA(20).
- Exit: Long exits when price crosses below Donchian(20) lower band; Short exits when price crosses above Donchian(20) upper band.
- Signal size: 0.25 discrete to balance capture and fee control.
- Works in bull (buying breakouts in uptrend) and bear (selling breakdowns in downtrend) with reduced whipsaws.
- Uses proper MTF alignment to avoid look-ahead.
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
    
    # Calculate Donchian(20) bands on 1d
    donchian_period = 20
    upper = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 1d timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Get 1d data for volume MA(20)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(donchian_period, 50, 20)  # Donchian needs 20, EMA50 needs 50, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(upper[i]) or 
            np.isnan(lower[i]) or 
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
                # Long: price breaks above Donchian upper band AND price > 1w EMA50 (uptrend)
                if curr_close > upper[i] and curr_close > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below Donchian lower band AND price < 1w EMA50 (downtrend)
                elif curr_close < lower[i] and curr_close < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit when price crosses below Donchian lower band
            if curr_close < lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price crosses above Donchian upper band
            if curr_close > upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0