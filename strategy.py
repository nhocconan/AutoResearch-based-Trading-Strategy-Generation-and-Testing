# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w EMA(34) trend filter and 1d Donchian(20) breakout with volume confirmation.
# Long when price breaks above 1d Donchian high in uptrend with volume surge.
# Short when price breaks below 1d Donchian low in downtrend with volume surge.
# Uses 1w EMA(34) for trend direction and 1d ATR(14) for dynamic breakout levels.
# Designed for low trade frequency (7-25/year) to minimize fee discount and capture sustained moves.

name = "1d_DonchianBreakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # 1w EMA(34) for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up = ema_34_1w[1:] > ema_34_1w[:-1]  # Rising EMA = uptrend
    trend_up = np.concatenate([[False], trend_up])  # Align with 1w index
    
    # 1d Donchian(20) breakout levels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d ATR(14) for dynamic breakout buffer
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    high_close[0] = high[0] - close[0]  # First value
    low_close[0] = low[0] - close[0]    # First value
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Dynamic breakout levels with ATR buffer
    upper_break = high_20 + (atr_14 * 0.5)
    lower_break = low_20 - (atr_14 * 0.5)
    
    # Align 1w trend to 1d timeframe
    trend_up_aligned = align_htf_to_ltf(prices, df_1w, trend_up.astype(float))
    
    # Volume confirmation: 1d volume > 2.0x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure enough data for EMA(34) and Donchian(20)
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(trend_up_aligned[i]) or np.isnan(upper_break[i]) or
            np.isnan(lower_break[i]) or np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long setup: break above Donchian high + ATR in uptrend with volume
            if (trend_up_aligned[i] > 0.5 and  # 1w uptrend
                close[i] > upper_break[i] and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short setup: break below Donchian low - ATR in downtrend with volume
            elif (trend_up_aligned[i] <= 0.5 and  # 1w downtrend
                  close[i] < lower_break[i] and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below Donchian low - ATR or trend turns down
            if close[i] < lower_break[i] or trend_up_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above Donchian high + ATR or trend turns up
            if close[i] > upper_break[i] or trend_up_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals