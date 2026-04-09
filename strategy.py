#!/usr/bin/env python3
# mtf_1h_ema_pullback_volume_4h1d_v1
# Hypothesis: 1h strategy trading EMA pullbacks in the direction of 4h/1d trend.
# Uses 4h EMA for trend direction and 1d EMA for higher timeframe filter.
# Entry: price pulls back to 21 EMA on 1h with volume confirmation during 08-20 UTC session.
# Exit: opposite signal or stop loss via EMA crossover. Designed for low trade frequency (15-35/year).
# Works in bull/bear by only trading with higher timeframe trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_ema_pullback_volume_4h1d_v1"
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
    
    # Pre-compute session hours filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    # 4h HTF data for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # 4h 21 EMA for trend
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # 1d HTF data for higher timeframe filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d 50 EMA for higher timeframe trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1h indicators for entry timing
    close_s = pd.Series(close)
    # 1h 21 EMA for pullback entries
    ema_21_1h = close_s.ewm(span=21, adjust=False, min_periods=21).mean().values
    # 1h 50 EMA for exit signals
    ema_50_1h = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(ema_21_1h[i]) or np.isnan(ema_50_1h[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC only
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.2x 20-period average
        volume_confirmed = volume[i] > 1.2 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below 50 EMA or volume dries up
            if close[i] < ema_50_1h[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price crosses above 50 EMA or volume dries up
            if close[i] > ema_50_1h[i] or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            if volume_confirmed:
                # Long entry: price pulls back to 21 EMA from above, 
                # 4h trend up (price > 4h 21 EMA), 1d trend up (price > 1d 50 EMA)
                if (close[i] >= ema_21_1h[i] * 0.998 and close[i] <= ema_21_1h[i] * 1.002 and
                    close[i] > ema_21_4h_aligned[i] and close[i] > ema_50_1d_aligned[i]):
                    position = 1
                    signals[i] = 0.20
                # Short entry: price pulls back to 21 EMA from below,
                # 4h trend down (price < 4h 21 EMA), 1d trend down (price < 1d 50 EMA)
                elif (close[i] >= ema_21_1h[i] * 0.998 and close[i] <= ema_21_1h[i] * 1.002 and
                      close[i] < ema_21_4h_aligned[i] and close[i] < ema_50_1d_aligned[i]):
                    position = -1
                    signals[i] = -0.20
    
    return signals