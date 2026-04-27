#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike
Hypothesis: Camarilla R1/S1 breakout on 4h with 12h EMA50 trend filter and volume confirmation.
Long when price breaks above R1 with 12h uptrend and volume spike.
Short when price breaks below S1 with 12h downtrend and volume spike.
Camarilla levels provide intraday support/resistance that work well in ranging markets.
Volume spike confirms breakout conviction. 12h trend filter avoids counter-trend trades.
Designed for 20-50 trades/year on 4h to minimize fee drag while maintaining edge in both bull and bear markets.
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
    
    # Calculate 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla levels for each 4h bar using previous 4h bar's OHLC
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.125*(high-low),
    # R2 = close + 0.75*(high-low), R1 = close + 0.5*(high-low),
    # S1 = close - 0.5*(high-low), S2 = close - 0.75*(high-low),
    # S3 = close - 1.125*(high-low), S4 = close - 1.5*(high-low)
    # We use R1 and S1 for breakout signals
    high_shift = np.roll(high, 1)
    low_shift = np.roll(low, 1)
    close_shift = np.roll(close, 1)
    high_shift[0] = np.nan
    low_shift[0] = np.nan
    close_shift[0] = np.nan
    
    camarilla_range = high_shift - low_shift
    r1 = close_shift + 0.5 * camarilla_range
    s1 = close_shift - 0.5 * camarilla_range
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25  # 25% position size
    
    # Warmup: need enough for previous bar data, EMA50, volume average
    start_idx = max(50, 20, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_trend = ema_50_12h_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Flat - look for entry: Camarilla R1/S1 breakout with 12h trend alignment and volume spike
            # Long: Close > R1 AND 12h trend up (close > EMA50) AND volume spike
            # Short: Close < S1 AND 12h trend down (close < EMA50) AND volume spike
            long_condition = close_val > r1[i] and close_val > ema_trend and vol_spike
            short_condition = close_val < s1[i] and close_val < ema_trend and vol_spike
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long - exit when price breaks below S1 (reversal) OR 12h trend turns down
            if close_val < s1[i] or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price breaks above R1 (reversal) OR 12h trend turns up
            if close_val > r1[i] or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0