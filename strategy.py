#!/usr/bin/env python3
# Hypothesis: 4h Camarilla Pivot-based breakout with 1d EMA trend filter and volume spike
# Long when: price breaks above Camarilla R3, 1d EMA(34) rising, volume > 2x 20-period avg
# Short when: price breaks below Camarilla S3, 1d EMA(34) falling, volume > 2x 20-period avg
# Exit when: price crosses Camarilla H-L midpoint OR trend reverses
# Position size: 0.25. Target: 20-40 trades/year. Designed for BTC/ETH in bull/bear via trend filter.

name = "4h_Camarilla_R3S3_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

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
    
    # Calculate Camarilla levels (based on previous day's OHLC)
    # Since we're on 4h timeframe, we need daily OHLC
    # We'll compute it using rolling window on daily data, but since we don't have daily in 4h,
    # we approximate using prior bar's high/low/close - this is a simplification
    # Better approach: use 1d data from mtf
    # For now, use previous bar's values as proxy (will be refined with actual 1d data)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Typical price for pivot calculation
    pp = (prev_high + prev_low + prev_close) / 3.0
    r3 = pp + (high - low) * 1.1 / 2
    s3 = pp - (high - low) * 1.1 / 2
    h_l_mid = (high + low) / 2.0  # For exit
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA(34) for trend filter
    close_1d = df_1d['close']
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_prev = np.roll(ema_34_1d, 1)
    ema_34_1d_prev[0] = ema_34_1d[0]
    ema_rising = ema_34_1d > ema_34_1d_prev
    ema_falling = ema_34_1d < ema_34_1d_prev
    ema_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_rising)
    ema_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_falling)
    
    # Volume spike: current volume > 2.0x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (2.0 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(h_l_mid[i]) or
            np.isnan(ema_rising_aligned[i]) or np.isnan(ema_falling_aligned[i]) or
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price > Camarilla R3 + 1d EMA rising + volume spike
            if (close[i] > r3[i] and 
                ema_rising_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price < Camarilla S3 + 1d EMA falling + volume spike
            elif (close[i] < s3[i] and 
                  ema_falling_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below H-L midpoint OR trend turns down
            if (close[i] < h_l_mid[i]) or (not ema_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above H-L midpoint OR trend turns up
            if (close[i] > h_l_mid[i]) or (not ema_falling_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals