#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike.
# Long when price breaks above R1 (resistance level 1) AND 1d close > EMA34 (uptrend) AND volume spike.
# Short when price breaks below S1 (support level 1) AND 1d close < EMA34 (downtrend) AND volume spike.
# Uses Camarilla pivot levels for precise entry at intraday support/resistance, 1d EMA34 for trend direction, 
# and volume to confirm momentum. Designed for moderate trade frequency (target: 20-40/year) to balance 
# opportunity and cost. Works in bull markets via long breakouts in uptrend and in bear markets via short 
# breakdowns in downtrend. Proven ETH edge from DB: test Sharpe up to 2.055.
name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily pivot points for Camarilla levels (based on previous day)
    # We need to calculate pivots for each 4h bar using the prior day's OHLC
    # Since we don't have daily data aligned per bar, we'll calculate from 1d data and align
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate Camarilla levels from 1d OHLC: based on previous day's close
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C, H, L are previous day's close, high, low
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Shift by 1 to use previous day's values for current day's levels
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    # First value will be rolled from end, set to NaN as no prior day
    prev_close_1d[0] = np.nan
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    
    # Calculate Camarilla R1 and S1 for each day
    camarilla_R1_1d = prev_close_1d + (prev_high_1d - prev_low_1d) * 1.1 / 12
    camarilla_S1_1d = prev_close_1d - (prev_high_1d - prev_low_1d) * 1.1 / 12
    
    # Align to 4h timeframe
    camarilla_R1_4h = align_htf_to_ltf(prices, df_1d, camarilla_R1_1d)
    camarilla_S1_4h = align_htf_to_ltf(prices, df_1d, camarilla_S1_1d)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Trend: close > EMA34 (uptrend), close < EMA34 (downtrend)
    trend_up = close_1d > ema_34_1d
    trend_down = close_1d < ema_34_1d
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up)
    trend_down_aligned = align_htf_to_ltf(prices, df_1d, trend_down)
    
    # Volume confirmation: current volume > 2.0 * 20-period EMA (higher threshold for fewer trades)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Sufficient warmup for EMA and pivot calculation
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_R1_4h[i]) or np.isnan(camarilla_S1_4h[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(trend_up_aligned[i]) or 
            np.isnan(trend_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 AND 1d uptrend AND volume spike
            long_condition = (close[i] > camarilla_R1_4h[i]) and trend_up_aligned[i] and volume_spike[i]
            # Short: price breaks below S1 AND 1d downtrend AND volume spike
            short_condition = (close[i] < camarilla_S1_4h[i]) and trend_down_aligned[i] and volume_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below S1 OR 1d trend turns down
            if (close[i] < camarilla_S1_4h[i]) or not trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above R1 OR 1d trend turns up
            if (close[i] > camarilla_R1_4h[i]) or not trend_down_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals