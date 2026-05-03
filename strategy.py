#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume confirmation.
# Long when price breaks above R1 in bull trend (close > 4h EMA50) with volume > 1.8x 20-period MA.
# Short when price breaks below S1 in bear trend (close < 4h EMA50) with volume spike.
# Uses 1h primary timeframe with 4h HTF for trend filter and Camarilla levels. Discrete sizing 0.20.
# Session filter (08-20 UTC) reduces noise. Target: 60-150 total trades over 4 years (15-37/year).
# Camarilla R1/S1 provide intraday support/resistance; 4h EMA50 filters counter-trend whipsaw.
# Works in bull (breakouts with trend) and bear (fades at resistance in downtrend).

name = "1h_Camarilla_R1S1_4hEMA50_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Camarilla levels and EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 4h Camarilla levels (based on previous 4h bar)
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    c_4h = df_4h['close'].values
    camarilla_r1 = c_4h + (h_4h - l_4h) * 1.1 / 12
    camarilla_s1 = c_4h - (h_4h - l_4h) * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe (wait for completed 4h bar)
    r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # Volume regime: current 1h volume > 1.8x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma_20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_4h_aligned[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Entry logic
        if position == 0:
            if is_bull_trend and close_val > r1 and vol_spike:
                signals[i] = 0.20
                position = 1
            elif is_bear_trend and close_val < s1 and vol_spike:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price breaks below S1 OR trend reversal
            if close_val < s1 or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above R1 OR trend reversal
            if close_val > r1 or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals