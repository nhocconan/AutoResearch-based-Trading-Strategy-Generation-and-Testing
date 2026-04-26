#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wEMA50_Trend_VolumeSpike
Hypothesis: On daily timeframe, Camarilla R1/S1 breakouts filtered by weekly EMA50 trend and volume spike capture institutional moves while minimizing trades. Long when price breaks above R1 in bullish weekly trend with volume confirmation; short when price breaks below S1 in bearish weekly trend with volume confirmation. Uses discrete sizing (±0.30) and targets 7-25 trades/year. Works in both bull/bear markets by only trading in direction of higher-timeframe (weekly) trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for higher-timeframe trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate previous day's Camarilla levels (using daily data from prices)
    # We need to compute daily OHLC from the 15m/1h/etc data, but since we're on 1d timeframe,
    # the prices DataFrame already contains daily bars
    if len(prices) < 2:
        return np.zeros(n)
    
    # Previous day's high, low, close for Camarilla calculation
    prev_high = prices['high'].shift(1).values  # Shift to get previous day
    prev_low = prices['low'].shift(1).values
    prev_close = prices['close'].shift(1).values
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_range = prev_high - prev_low
    r1 = prev_close + camarilla_range * 1.1 / 12
    s1 = prev_close - camarilla_range * 1.1 / 12
    
    # Volume confirmation: volume > 2.0x 20-period average (stricter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.30
    
    # Warmup: max of calculations (20 for volume MA, 1 for shift, 50 for EMA)
    start_idx = max(20, 1, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready (NaN from calculation)
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        r1_val = r1[i]
        s1_val = s1[i]
        ema_50_val = ema_50_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine weekly trend: bullish if price > EMA50, bearish if price < EMA50
        bullish_1w = close_val > ema_50_val
        bearish_1w = close_val < ema_50_val
        
        # Entry conditions: price breaks above/below Camarilla levels in direction of weekly trend with volume confirmation
        long_entry = (close_val > r1_val) and bullish_1w and vol_spike
        short_entry = (close_val < s1_val) and bearish_1w and vol_spike
        
        # Exit conditions: price returns inside Camarilla levels or trend reversal
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and (close_val < r1_val or not bullish_1w):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close_val > s1_val or not bearish_1w):
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0