#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe strategy using 4h trend filter (EMA50) and 1d Camarilla pivot breakouts with volume confirmation.
# Uses 4h for trend direction, 1d for key support/resistance levels, and 1h for precise entry timing.
# Volume spike filter reduces false breakouts. Session filter (08-20 UTC) avoids low-liquidity hours.
# Designed to work in both bull and bear markets by following 4h trend and fading false breaks at key levels.
# Target: 15-30 trades/year to avoid fee drag.

name = "1h_Camarilla_R1S1_4hTrend_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) - avoid low liquidity periods
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivot levels (daily)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate 50-period EMA on 4h close for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla pivot levels from previous day's OHLC
    prev_day_high = df_1d['high'].shift(1).values
    prev_day_low = df_1d['low'].shift(1).values
    prev_day_close = df_1d['close'].shift(1).values
    
    # Camarilla formula for R1 and S1 only (most significant levels)
    range_ = prev_day_high - prev_day_low
    camarilla_mult = 1.1 / 12  # ~0.0916667
    r1 = prev_day_close + range_ * camarilla_mult * 1
    s1 = prev_day_close - range_ * camarilla_mult * 1
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate 24-period volume average for spike detection (reduces noise)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24)  # Need 50 for 4h EMA and 24 for volume average
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_4h = ema_50_4h_aligned[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        vol = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Enter long: Price breaks above R1 with volume AND price > 4h EMA50 (uptrend)
            if close[i] > r1_level and vol > 2.0 * vol_ma_val and close[i] > ema_4h:
                signals[i] = 0.20
                position = 1
            # Enter short: Price breaks below S1 with volume AND price < 4h EMA50 (downtrend)
            elif close[i] < s1_level and vol > 2.0 * vol_ma_val and close[i] < ema_4h:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: Price breaks below R1 OR trend reverses (price < 4h EMA50)
            if close[i] < r1_level or close[i] < ema_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: Price breaks above S1 OR trend reverses (price > 4h EMA50)
            if close[i] > s1_level or close[i] > ema_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals