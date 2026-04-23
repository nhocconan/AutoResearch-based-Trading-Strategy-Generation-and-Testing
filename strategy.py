#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R1/S1 breakout + 4h EMA(34) trend filter + volume confirmation (>1.5x 20-period average) + session filter (08-20 UTC)
- Camarilla pivot points identify intraday support/resistance levels where breakouts often occur
- 4h EMA(34) ensures trades align with higher-timeframe trend to avoid counter-trend whipsaws
- Volume confirmation (>1.5x average) validates breakout strength
- Session filter (08-20 UTC) avoids low-liquidity Asian session noise
- Designed for 1h timeframe targeting 15-37 trades/year (60-150 over 4 years)
- Works in both bull and bear markets by trading with 4h trend from intraday breakouts
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Precompute session filter
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot points (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla equations: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe (1-day delay for previous day's data)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1, additional_delay_bars=1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1, additional_delay_bars=1)
    
    # Get 4h data for EMA(34) trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA(34) for trend filter
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_ma[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Camarilla breakout conditions
        # Long: price breaks above R1 level
        # Short: price breaks below S1 level
        long_breakout = close[i] > camarilla_r1_aligned[i]
        short_breakout = close[i] < camarilla_s1_aligned[i]
        
        # Trend filter: price > EMA for long, price < EMA for short
        uptrend = close[i] > ema_34_4h_aligned[i]
        downtrend = close[i] < ema_34_4h_aligned[i]
        
        if position == 0:
            # Long conditions: breakout above R1, uptrend, volume spike, in session
            long_signal = (long_breakout and 
                          uptrend and
                          volume[i] > 1.5 * vol_ma[i])
            
            # Short conditions: breakout below S1, downtrend, volume spike, in session
            short_signal = (short_breakout and 
                           downtrend and
                           volume[i] > 1.5 * vol_ma[i])
            
            if long_signal:
                signals[i] = 0.20
                position = 1
            elif short_signal:
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions: price returns to opposite Camarilla level or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: price falls below S1 level or trend turns down
                if (close[i] < camarilla_s1_aligned[i] or 
                    not uptrend):  # Trend reversal
                    exit_signal = True
            elif position == -1:
                # Exit short: price rises above R1 level or trend turns up
                if (close[i] > camarilla_r1_aligned[i] or 
                    not downtrend):  # Trend reversal
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hEMA34_Trend_VolumeConfirm_Session"
timeframe = "1h"
leverage = 1.0