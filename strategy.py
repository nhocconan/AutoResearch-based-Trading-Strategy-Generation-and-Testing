#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above Camarilla R1 AND close > EMA34(1d) AND volume > 2.0x 20-period average
# Short when price breaks below Camarilla S1 AND close < EMA34(1d) AND volume > 2.0x 20-period average
# Exit when price retraces to Camarilla pivot point OR close crosses EMA34(1d) (trend flip)
# Uses 4h primary timeframe with 1d HTF for trend filter to capture institutional participation
# Discrete sizing (0.25) to limit fee drag and manage drawdown in both bull and bear markets
# Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag
# Camarilla levels from daily timeframe provide structure; volume confirmation avoids false breakouts

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for EMA34 trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d OHLC (using completed daily bar)
    # Camarilla: Pivot = (H+L+C)/3, Range = H-L
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    prev_high = df_1d['high'].shift(1).values  # Previous day's high
    prev_low = df_1d['low'].shift(1).values    # Previous day's low
    prev_close = df_1d['close'].shift(1).values # Previous day's close
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    r1 = prev_close + (range_hl * 1.1 / 12.0)
    s1 = prev_close - (range_hl * 1.1 / 12.0)
    
    # Align Camarilla levels to 4h timeframe (wait for completed 1d bar)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: volume > 2.0x 20-period average (strict to reduce trades)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R1 AND close > EMA34(1d) AND volume spike
            if (high[i] > r1_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S1 AND close < EMA34(1d) AND volume spike
            elif (low[i] < s1_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retracement to pivot point OR close < EMA34(1d) (trend flip)
            if close[i] <= pivot_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retracement to pivot point OR close > EMA34(1d) (trend flip)
            if close[i] >= pivot_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals