#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla Pivot Reversal with Daily EMA34 Trend Filter and Volume Spike
- Uses 6h Camarilla pivot levels (R3/S3 for reversal, R4/S4 for breakout) calculated from prior 1d candle
- Higher timeframe trend filter: 1d EMA34 - only trade reversals in direction of daily trend
- Volume confirmation (> 2.0x 24-period average) ensures institutional participation
- Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Works in bull markets (fade at S3/R3 in uptrend) and bear markets (fade at R3/S3 in downtrend)
- Avoids overtrading by requiring confluence of pivot level, trend alignment, and volume spike
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get prior 1d data for Camarilla pivot calculation (use previous completed 1d bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivots from prior 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    R3 = pivot + range_1d * 1.1 / 4.0
    S3 = pivot - range_1d * 1.1 / 4.0
    R4 = pivot + range_1d * 1.1 / 2.0
    S4 = pivot - range_1d * 1.1 / 2.0
    
    # Align pivots to 6h timeframe (use prior 1d bar's levels)
    R3_6h = align_htf_to_ltf(prices, df_1d, R3)
    S3_6h = align_htf_to_ltf(prices, df_1d, S3)
    R4_6h = align_htf_to_ltf(prices, df_1d, R4)
    S4_6h = align_htf_to_ltf(prices, df_1d, S4)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_6h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: > 2.0x 24-period average (4h equivalent on 6h chart)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 24)  # for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R3_6h[i]) or np.isnan(S3_6h[i]) or np.isnan(ema_34_6h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price at S3/S4 with bullish daily trend and volume spike
            long_at_S3 = low[i] <= S3_6h[i] * 1.001  # Allow small penetration
            long_at_S4 = low[i] <= S4_6h[i] * 1.001
            bullish_trend = close[i] > ema_34_6h[i]
            volume_spike = volume[i] > 2.0 * vol_ma[i]
            
            if ((long_at_S3 or long_at_S4) and bullish_trend and volume_spike):
                signals[i] = 0.25
                position = 1
            
            # Short conditions: price at R3/R4 with bearish daily trend and volume spike
            elif not ((long_at_S3 or long_at_S4) and bullish_trend and volume_spike):
                short_at_R3 = high[i] >= R3_6h[i] * 0.999  # Allow small penetration
                short_at_R4 = high[i] >= R4_6h[i] * 0.999
                bearish_trend = close[i] < ema_34_6h[i]
                
                if ((short_at_R3 or short_at_R4) and bearish_trend and volume_spike):
                    signals[i] = -0.25
                    position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reaches pivot or daily trend turns bearish
                if high[i] >= pivot[i] * 0.999 or close[i] < ema_34_6h[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: price reaches pivot or daily trend turns bullish
                if low[i] <= pivot[i] * 1.001 or close[i] > ema_34_6h[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Camarilla_R3S4_Reversal_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0