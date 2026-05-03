#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume confirmation.
# Long when price breaks above Camarilla R1 level in bull trend (close > 4h EMA50) with volume > 1.8x 20-period MA.
# Short when price breaks below Camarilla S1 level in bear trend (close < 4h EMA50) with volume spike.
# Uses discrete position sizing (0.20) to minimize fee churn. 4h EMA50 provides responsive trend filter.
# Volume confirmation ensures institutional participation. Session filter (08-20 UTC) reduces noise trades.
# Target: 80-120 total trades over 4 years = 20-30/year for 1h.

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA50 trend filter and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels (based on previous 4h bar's range)
    prev_4h_high = df_4h['high'].values
    prev_4h_low = df_4h['low'].values
    prev_4h_close = df_4h['close'].values
    
    # Calculate Camarilla R1 and S1 levels for each 4h bar
    camarilla_r1 = prev_4h_close + (prev_4h_high - prev_4h_low) * 1.1 / 12
    camarilla_s1 = prev_4h_close - (prev_4h_high - prev_4h_low) * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe (no additional delay needed as these are based on completed 4h bar)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # Volume regime: current 1h volume > 1.8x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Session filter: only trade between 08-20 UTC
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_4h_aligned[i]
        r1_level = camarilla_r1_aligned[i]
        s1_level = camarilla_s1_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Camarilla breakout conditions
        breakout_r1 = close_val > r1_level
        breakout_s1 = close_val < s1_level
        
        # Entry logic
        if position == 0:
            if is_bull_trend and breakout_r1 and vol_spike:
                signals[i] = 0.20
                position = 1
            elif is_bear_trend and breakout_s1 and vol_spike:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price breaks below S1 level OR trend reversal
            if close_val < s1_level or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above R1 level OR trend reversal
            if close_val > r1_level or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals