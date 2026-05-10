#!/usr/bin/env python3
# 1h_4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Use 4h Camarilla R1/S1 levels for breakout signals, 1d EMA34 for trend filter, and volume confirmation.
# Trades only during 08-20 UTC to reduce noise. Position size fixed at 0.20 to limit drawdown.
# Works in bull markets by buying breakouts in uptrends, and in bear markets by selling breakdowns in downtrends.

name = "1h_4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "1h"
leverage = 1.0

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
    
    # Get 4h data for Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h Camarilla levels (based on previous 4h bar)
    # Typical Price = (H + L + C) / 3
    # Range = H - L
    # R1 = Close + 1.1 * (Range) / 12
    # S1 = Close - 1.1 * (Range) / 12
    fourh_close = df_4h['close'].values
    fourh_high = df_4h['high'].values
    fourh_low = df_4h['low'].values
    
    typical_price = (fourh_high + fourh_low + fourh_close) / 3
    range_hl = fourh_high - fourh_low
    camarilla_r1 = fourh_close + 1.1 * range_hl / 12
    camarilla_s1 = fourh_close - 1.1 * range_hl / 12
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # Volume confirmation (24-period MA on 1h = 1 day)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need daily EMA34 (34), volume MA (24)
    start_idx = max(34, 24)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Daily trend filter
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            if in_session:
                # Long entry: uptrend + price breaks above Camarilla R1 + volume
                if uptrend and close[i] > camarilla_r1_aligned[i] and volume_confirm:
                    signals[i] = 0.20
                    position = 1
                # Short entry: downtrend + price breaks below Camarilla S1 + volume
                elif downtrend and close[i] < camarilla_s1_aligned[i] and volume_confirm:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long exit: trend breaks or price re-enters below R1
            if not uptrend or close[i] < camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: trend breaks or price re-enters above S1
            if not downtrend or close[i] > camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals