#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume spike confirmation.
# Long when price breaks above Camarilla R1 level AND 12h close > 12h EMA50 (uptrend) AND 4h volume > 1.8x 20-period volume MA.
# Short when price breaks below Camarilla S1 level AND 12h close < 12h EMA50 (downtrend) AND 4h volume > 1.8x 20-period volume MA.
# Exit on retracement to Camarilla pivot point (PP) or trend reversal.
# Uses session filter (08-20 UTC) to avoid low-liquidity periods. Position size 0.25.
# Designed for 4h timeframe to achieve 75-200 total trades over 4 years (19-50/year) with strict entry conditions.
# Camarilla levels provide objective support/resistance, 12h EMA50 filters for higher-timeframe trend alignment, volume confirms participation.
# Works in both bull and bear markets by only trading breakouts in the direction of the 12h trend when volume confirms.

name = "4h_Camarilla_R1S1_Breakout_12hEMA50_VolumeSpike_Session"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 12h data for trend filter and Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend direction
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla levels from previous 12h bar (OHLC)
    # Camarilla: PP = (H+L+C)/3, R1 = PP + (H-L)*1.1/12, S1 = PP - (H-L)*1.1/12
    lookback = 1
    if len(df_12h) < lookback + 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 12h bar
    camarilla_pp = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3
    camarilla_range = df_12h['high'] - df_12h['low']
    camarilla_r1 = camarilla_pp + camarilla_range * 1.1 / 12
    camarilla_s1 = camarilla_pp - camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_12h, camarilla_pp.values)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1.values)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1.values)
    
    # Calculate 4h volume 20-period MA for spike detection
    volume_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(camarilla_pp_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(volume_ma_4h[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: current 4h volume > 1.8x 20-period volume MA
        volume_spike = volume[i] > (volume_ma_4h[i] * 1.8)
        
        # Camarilla breakout conditions
        breakout_up = high_val > camarilla_r1_aligned[i]  # Price breaks above R1
        breakout_down = low_val < camarilla_s1_aligned[i]  # Price breaks below S1
        
        # 12h trend conditions
        trend_up = close_val > ema_50_12h_aligned[i]   # 12h uptrend
        trend_down = close_val < ema_50_12h_aligned[i]  # 12h downtrend
        
        if position == 0:
            # Long: Camarilla breakout up AND 12h uptrend AND volume spike AND session
            if breakout_up and trend_up and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Camarilla breakout down AND 12h downtrend AND volume spike AND session
            elif breakout_down and trend_down and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retouches Camarilla PP OR trend changes
            if close_val < camarilla_pp_aligned[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retouches Camarilla PP OR trend changes
            if close_val > camarilla_pp_aligned[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals