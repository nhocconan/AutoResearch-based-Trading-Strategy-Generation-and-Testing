#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hEMA50_1dTrend_VolumeSpike_v1
Hypothesis: 1-hour Camarilla R1/S1 breakout with 4-hour EMA50 and daily trend filters plus volume confirmation.
Uses 4h/1d for signal direction and 1h for precise entry timing. Discrete position sizing (0.20) to minimize fee drag.
Session filter (08-20 UTC) reduces noise trades. Designed for 15-30 trades/year to overcome 1h timeframe challenges.
Works in bull markets via breakouts with trend alignment and in bear markets via short opportunities with volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) - open_time is already datetime64[ms]
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for HTF filters (EMA50 trend)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate EMA(50) on 4h for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA(34) on daily for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume spike filter: volume > 2.0 * 20-period average (stricter for 1h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Calculate Camarilla levels from previous 1h bar
    prev_high = prices['high'].shift(1).values
    prev_low = prices['low'].shift(1).values
    prev_close = prices['close'].shift(1).values
    
    # Camarilla R1 = C + ((H-L)*1.1/12), S1 = C - ((H-L)*1.1/12)
    camarilla_r1 = prev_close + ((prev_high - prev_low) * 1.1 / 12)
    camarilla_s1 = prev_close - ((prev_high - prev_low) * 1.1 / 12)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 4h EMA(50), 1d EMA(34), volume MA
    start_idx = max(50, 34, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
            
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        close_val = close[i]
        trend_4h_up = close_val > ema_50_4h_aligned[i]   # 4h uptrend
        trend_4h_down = close_val < ema_50_4h_aligned[i]  # 4h downtrend
        trend_1d_up = close_val > ema_34_1d_aligned[i]   # Daily uptrend
        trend_1d_down = close_val < ema_34_1d_aligned[i]  # Daily downtrend
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R1 AND 4h trend up AND daily trend up AND volume spike
            long_signal = (close_val > camarilla_r1[i]) and trend_4h_up and trend_1d_up and vol_spike
            
            # Short: price breaks below Camarilla S1 AND 4h trend down AND daily trend down AND volume spike
            short_signal = (close_val < camarilla_s1[i]) and trend_4h_down and trend_1d_down and vol_spike
            
            if long_signal:
                signals[i] = 0.20
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.20
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: 4h trend flips down OR daily trend flips down
            if (not trend_4h_up) or (not trend_1d_up):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: 4h trend flips up OR daily trend flips up
            if (not trend_4h_down) or (not trend_1d_down):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_1dTrend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0