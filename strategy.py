#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R1S1 Breakout with 4h EMA34 Trend and Volume Confirmation + Session Filter (08-20 UTC)
- Camarilla R1/S1 levels provide precise intraday breakout/reversal points from daily extremes
- 4h EMA(34) ensures alignment with higher timeframe trend for multi-timeframe confirmation
- Volume > 1.5x 20-period average confirms breakout strength and reduces false signals
- Session filter (08-20 UTC) avoids low-liquidity Asian session noise
- Designed for 1h timeframe targeting 15-35 trades/year (60-140 over 4 years) to minimize fee drag
- Works in bull markets via breakouts with trend, in bear markets via fade of overextended moves
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) - open_time is already datetime64[ms]
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate Camarilla levels from previous day (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values  # shift(1) to use previous day's data
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla R1, S1 levels: R1 = close + (high-low)*1.1/12, S1 = close - (high-low)*1.1/12
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe (completed 1d bar only)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA4h, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Camarilla breakout signals with trend filter and volume confirmation
        # Long: price breaks above Camarilla R1 + uptrend + volume spike
        # Short: price breaks below Camarilla S1 + downtrend + volume spike
        long_signal = (close[i] > camarilla_r1_aligned[i] and 
                      close[i] > ema_34_4h_aligned[i] and
                      volume[i] > 1.5 * vol_ma[i])
        
        short_signal = (close[i] < camarilla_s1_aligned[i] and 
                       close[i] < ema_34_4h_aligned[i] and
                       volume[i] > 1.5 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.20
                position = 1
            elif short_signal:
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions: trend reversal or opposite Camarilla level break
            exit_signal = False
            
            if position == 1:
                # Exit long: trend reversal or price breaks below Camarilla S1
                if (close[i] < ema_34_4h_aligned[i] or 
                    close[i] < camarilla_s1_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: trend reversal or price breaks above Camarilla R1
                if (close[i] > ema_34_4h_aligned[i] or 
                    close[i] > camarilla_r1_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hEMA34_Trend_VolumeSession"
timeframe = "1h"
leverage = 1.0