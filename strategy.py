#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 Breakout + 1d EMA34 Trend Filter + Volume Spike Confirmation
# Camarilla pivot levels (R3/S3) act as strong support/resistance on 1d timeframe.
# Breakout above R3 or below S3 with volume spike indicates institutional interest.
# 1d EMA34 ensures alignment with daily trend to avoid counter-trend trades.
# Designed for 12-37 trades/year on 12h timeframe to minimize fee drag while capturing strong trends.
# Works in bull markets via long breakouts in uptrend and bear markets via short breakdowns in downtrend.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend filter and Camarilla levels - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    # Camarilla equations:
    # R4 = close + ((high - low) * 1.1/2)
    # R3 = close + ((high - low) * 1.1/4)
    # R2 = close + ((high - low) * 1.1/6)
    # R1 = close + ((high - low) * 1.1/12)
    # PP = (high + low + close) / 3
    # S1 = close - ((high - low) * 1.1/12)
    # S2 = close - ((high - low) * 1.1/6)
    # S3 = close - ((high - low) * 1.1/4)
    # S4 = close - ((high - low) * 1.1/2)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_r3 = close_1d + ((high_1d - low_1d) * 1.1 / 4)
    camarilla_s3 = close_1d - ((high_1d - low_1d) * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe (wait for 1d bar to close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate volume spike filter (20-period volume EMA on 12h)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 2.0)  # Volume at least 2x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND 1d uptrend AND volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_34_aligned[i] and  # 1d uptrend
                volume_spike[i]):
                signals[i] = 0.30
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND 1d downtrend AND volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_34_aligned[i] and  # 1d downtrend
                  volume_spike[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price closes below Camarilla PP (pivot point) OR 1d trend turns down
            camarilla_pp = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3
            camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
            if (close[i] < camarilla_pp_aligned[i] or 
                close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price closes above Camarilla PP OR 1d trend turns up
            camarilla_pp = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3
            camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
            if (close[i] > camarilla_pp_aligned[i] or 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals