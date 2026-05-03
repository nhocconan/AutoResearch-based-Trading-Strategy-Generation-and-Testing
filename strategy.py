#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume confirmation.
# Long when price breaks above Camarilla R1 AND 4h close > 4h EMA50 (uptrend) AND 1h volume > 2.0x 20-period volume MA.
# Short when price breaks below Camarilla S1 AND 4h close < 4h EMA50 (downtrend) AND 1h volume > 2.0x 20-period volume MA.
# Exit on retracement to Camarilla pivot point (PP) or trend reversal.
# Uses session filter (08-20 UTC) to avoid low-liquidity periods. Position size 0.20.
# Designed for 1h timeframe to achieve 60-150 total trades over 4 years (15-37/year) with strict entry conditions.
# Camarilla levels from 1h provide intraday support/resistance, 4h EMA50 filters for trend alignment, volume confirms participation.
# Works in both bull and bear markets by only trading breakouts in the direction of the 4h trend when volume confirms.

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_VolumeSpike_Session"
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
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend direction
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels from previous 1h bar (OHLC)
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    prev_close = df_4h['close'].shift(1).values
    
    # Calculate Camarilla R1 and S1 levels
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Calculate Camarilla pivot point (PP) for exit
    camarilla_pp = (prev_high + prev_low + prev_close) / 3
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pp)
    
    # Calculate 1h volume 20-period MA for spike detection
    volume_ma_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(camarilla_pp_aligned[i]) or 
            np.isnan(volume_ma_1h[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: current 1h volume > 2.0x 20-period volume MA
        volume_spike = volume[i] > (volume_ma_1h[i] * 2.0)
        
        # Camarilla breakout conditions
        breakout_up = high_val > camarilla_r1_aligned[i]  # Price breaks above R1
        breakout_down = low_val < camarilla_s1_aligned[i]  # Price breaks below S1
        
        # 4h trend conditions
        trend_up = close_val > ema_50_4h_aligned[i]   # 4h uptrend
        trend_down = close_val < ema_50_4h_aligned[i]  # 4h downtrend
        
        if position == 0:
            # Long: Camarilla breakout up AND 4h uptrend AND volume spike AND session
            if breakout_up and trend_up and volume_spike:
                signals[i] = 0.20
                position = 1
            # Short: Camarilla breakout down AND 4h downtrend AND volume spike AND session
            elif breakout_down and trend_down and volume_spike:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price retouches Camarilla pivot point (PP) OR trend changes
            if close_val < camarilla_pp_aligned[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price retouches Camarilla pivot point (PP) OR trend changes
            if close_val > camarilla_pp_aligned[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals