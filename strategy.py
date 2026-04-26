#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike
Hypothesis: On 1h timeframe, Camarilla R1/S1 breakouts filtered by 4h EMA50 trend and volume spikes capture institutional momentum with controlled trade frequency. Long when price breaks above R1 in bullish 4h trend with volume confirmation; short when price breaks below S1 in bearish 4h trend with volume confirmation. Uses discrete sizing (±0.20) and session filter (08-20 UTC) to target 15-35 trades/year. Works in bull/bear markets by only trading in direction of 4h trend.
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
    
    # Load 4h data ONCE before loop for trend and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for higher-timeframe trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels from previous 4h bar's OHLC
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_prev = df_4h['close'].shift(1).values  # Previous bar close
    
    # Camarilla R1, S1 levels (using previous 4h bar)
    # R1 = Close + 1.1*(High-Low)/12
    # S1 = Close - 1.1*(High-Low)/12
    camarilla_range = high_4h - low_4h
    r1 = close_4h_prev + 1.1 * camarilla_range / 12
    s1 = close_4h_prev - 1.1 * camarilla_range / 12
    
    # Align Camarilla levels to 1h timeframe (wait for 4h bar close)
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    
    # Volume spike filter: 1h volume > 1.5x 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    # Session filter: 08-20 UTC (pre-compute hours array)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.20
    
    # Warmup: max of EMA50 (50), volume EMA (20), Camarilla calculation (need 1 previous 4h bar)
    start_idx = 50  # Ensure 4h EMA50 and volume EMA are ready
    
    for i in range(start_idx, n):
        # Skip if any data not ready (NaN from calculation)
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i])):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_50_val = ema_50_4h_aligned[i]
        vol_spike = volume_spike[i]
        session = in_session[i]
        
        # Determine 4h trend: bullish if price > EMA50, bearish if price < EMA50
        bullish_4h = close_val > ema_50_val
        bearish_4h = close_val < ema_50_val
        
        # Entry conditions: price breaks R1/S1 with volume and session confirmation
        long_entry = (close_val > r1_val) and bullish_4h and vol_spike and session
        short_entry = (close_val < s1_val) and bearish_4h and vol_spike and session
        
        # Exit conditions: price returns inside Camarilla levels or trend reversal
        long_exit = (close_val < r1_val) or (close_val > s1_val) or not bullish_4h
        short_exit = (close_val > s1_val) or (close_val < r1_val) or not bearish_4h
        
        # Simplified exit: flip signal on opposite condition or level re-entry
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and (close_val < r1_val or not bullish_4h):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close_val > s1_val or not bearish_4h):
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike"
timeframe = "1h"
leverage = 1.0