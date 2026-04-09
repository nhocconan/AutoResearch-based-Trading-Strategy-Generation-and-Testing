#!/usr/bin/env python3
# 1h_ema_pullback_4h1d_volume_v1
# Hypothesis: 1h strategy using 4h EMA trend and 1d EMA filter for direction, with 1h EMA pullback entries and volume confirmation.
# In bull markets: 4h EMA up + 1d EMA up → long on 1h pullback to 21 EMA with volume spike.
# In bear markets: 4h EMA down + 1d EMA down → short on 1h pullback to 21 EMA with volume spike.
# Uses discrete sizing (0.0, ±0.20) to minimize fee churn. Session filter (08-20 UTC) reduces noise.
# Target: 60-150 total trades over 4 years = 15-37/year for 1h.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_ema_pullback_4h1d_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute hour filter for session (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h HTF data for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA21 for trend
    close_4h = pd.Series(df_4h['close'].values)
    ema_21_4h = close_4h.ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # 1d HTF data for higher timeframe filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for regime filter
    close_1d = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1h EMA21 for pullback entries
    close_s = pd.Series(close)
    ema_21_1h = close_s.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # 1h volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(ema_21_1h[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.8x 20-period average
        volume_confirmed = volume[i] > 1.8 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below 1h EMA21 or volume dries up
            if close[i] < ema_21_1h[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price closes above 1h EMA21 or volume dries up
            if close[i] > ema_21_1h[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            if volume_confirmed:
                # Long entry: 4h and 1d EMAs bullish + price pulls back to 1h EMA21
                if (ema_21_4h_aligned[i] > ema_21_4h_aligned[i-1] and  # 4h EMA rising
                    ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1] and  # 1d EMA rising
                    low[i] <= ema_21_1h[i] and close[i] > ema_21_1h[i]):  # Pullback to EMA with close above
                    position = 1
                    signals[i] = 0.20
                # Short entry: 4h and 1d EMAs bearish + price pulls back to 1h EMA21
                elif (ema_21_4h_aligned[i] < ema_21_4h_aligned[i-1] and  # 4h EMA falling
                      ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] and  # 1d EMA falling
                      high[i] >= ema_21_1h[i] and close[i] < ema_21_1h[i]):  # Pullback to EMA with close below
                    position = -1
                    signals[i] = -0.20
    
    return signals