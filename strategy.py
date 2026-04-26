#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeConfirm_v1
Hypothesis: On 1h timeframe, buy when price breaks above Camarilla R1 level with 4h uptrend (close > EMA34) and volume > 1.5x 20-period average; sell when price breaks below S1 level with 4h downtrend (close < EMA34) and volume > 1.5x average. Uses discrete sizing (0.0, ±0.20) to minimize fee churn. Targets 15-37 trades per year over 4 years by using 4h for signal direction and 1h only for entry timing, plus UTC 08-20 session filter to reduce noise.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for HTF trend filter (EMA34) and Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:  # need enough for EMA34
        return np.zeros(n)
    
    # Calculate 4h EMA34 for trend filter
    close_4h = pd.Series(df_4h['close'].values)
    ema_34_4h = close_4h.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate Camarilla levels from previous 4h bar
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C = (H+L+C)/3 (typical price)
    typical_price_4h = (df_4h['high'].values + df_4h['low'].values + df_4h['close'].values) / 3.0
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    camarilla_R1_4h = typical_price_4h + (high_4h - low_4h) * 1.1 / 12.0
    camarilla_S1_4h = typical_price_4h - (high_4h - low_4h) * 1.1 / 12.0
    
    # Align Camarilla levels to 1h timeframe (use previous 4h bar's levels)
    camarilla_R1_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_R1_4h)
    camarilla_S1_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_S1_4h)
    
    # Calculate volume average (20-period) for confirmation
    volume_s = pd.Series(volume)
    volume_ma_20 = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC) - open_time is already datetime64[ms]
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA warmup + volume MA warmup
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(camarilla_R1_4h_aligned[i]) or 
            np.isnan(camarilla_S1_4h_aligned[i]) or 
            np.isnan(volume_ma_20[i]) or 
            not in_session[i]):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * volume_ma_20[i]
        
        # 4h trend filter
        trend_uptrend = close[i] > ema_34_4h_aligned[i]
        trend_downtrend = close[i] < ema_34_4h_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 + volume confirm + 4h uptrend
            long_signal = (close[i] > camarilla_R1_4h_aligned[i] and 
                          volume_confirm and 
                          trend_uptrend)
            
            # Short: price breaks below S1 + volume confirm + 4h downtrend
            short_signal = (close[i] < camarilla_S1_4h_aligned[i] and 
                           volume_confirm and 
                           trend_downtrend)
            
            if long_signal:
                signals[i] = 0.20
                position = 1
            elif short_signal:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: price breaks below S1 (mean reversion) OR trend change to downtrend
            if close[i] < camarilla_S1_4h_aligned[i] or not trend_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: price breaks above R1 (mean reversion) OR trend change to uptrend
            if close[i] > camarilla_R1_4h_aligned[i] or not trend_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0