#!/usr/bin/env python3
# 4h_1d_1w_ema_touch_trend_breakout_v1
# Hypothesis: Price bouncing off key EMA levels (21/50) on 4h with volume confirmation and weekly trend filter.
# Long when price touches EMA21 from above with volume > 1.5x 20-period average and weekly close > weekly EMA50.
# Short when price touches EMA50 from below with volume > 1.5x 20-period average and weekly close < weekly EMA50.
# Uses 4h for entry timing and weekly trend filter to reduce whipsaw. Designed for 20-35 trades/year.
# Works in bull markets via EMA21 bounces and bear markets via EMA50 rejections.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_1w_ema_touch_trend_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h EMA21 and EMA50 for touch signals
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 4h volume MA(20) for volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    weekly_ema50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure EMA50 and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(ema21[i]) or np.isnan(ema50[i]) or np.isnan(vol_ma_20[i]) or np.isnan(weekly_ema50_aligned[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        # Price touch conditions with small buffer to avoid whipsaw
        touch_ema21_from_above = low[i] <= ema21[i] * 1.001 and high[i] >= ema21[i] * 0.999
        touch_ema50_from_below = high[i] >= ema50[i] * 0.999 and low[i] <= ema50[i] * 1.001
        
        if position == 1:  # Long position
            # Exit: price breaks below EMA50 or weekly trend turns bearish
            if close[i] < ema50[i] or weekly_ema50_aligned[i] < weekly_close[-1] if len(weekly_close) > 0 else False:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above EMA21 or weekly trend turns bullish
            if close[i] > ema21[i] or weekly_ema50_aligned[i] > weekly_close[-1] if len(weekly_close) > 0 else False:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price touches EMA21 from above with volume surge and weekly uptrend
            if touch_ema21_from_above and vol_surge and weekly_ema50_aligned[i] > weekly_close[-1] if len(weekly_close) > 0 else False:
                position = 1
                signals[i] = 0.25
            # Short entry: price touches EMA50 from below with volume surge and weekly downtrend
            elif touch_ema50_from_below and vol_surge and weekly_ema50_aligned[i] < weekly_close[-1] if len(weekly_close) > 0 else False:
                position = -1
                signals[i] = -0.25
    
    return signals