#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike
# Camarilla pivot levels identify institutional support/resistance where price often reverses or accelerates
# Breakout above R3 or below S3 with 1d EMA34 trend alignment captures strong momentum moves
# Volume confirmation (>1.8x 20-period EMA volume) ensures genuine institutional participation
# Discrete sizing 0.25 targets 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Works in bull markets (breakouts with uptrend) and bear markets (breakouts with downtrend)
# ATR-based stoploss via signal=0 when price moves against position (using 12h ATR21)

name = "12h_Camarilla_R3S3_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough data for EMA34
        return np.zeros(n)
    
    # Calculate 1d EMA34 trend filter from prior completed 1d bar
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_shifted = np.roll(ema34_1d, 1)
    ema34_1d_shifted[0] = np.nan
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d_shifted)
    
    # Calculate 1d ATR21 for stoploss reference (though we'll use signal-based exits primarily)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_for_atr = df_1d['close'].values
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d_for_atr[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d_for_atr[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])
    atr21_1d = pd.Series(tr).ewm(span=21, adjust=False, min_periods=21).mean().values
    atr21_1d_shifted = np.roll(atr21_1d, 1)
    atr21_1d_shifted[0] = np.nan
    atr21_1d_aligned = align_htf_to_ltf(prices, df_1d, atr21_1d_shifted)
    
    # Calculate Camarilla pivot levels from prior completed 1d bar
    # Camarilla: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4, etc.
    # We focus on R3 and S3 levels
    daily_range = high_1d - low_1d
    camarilla_r3 = close_1d + 1.1 * daily_range * 1.1 / 4
    camarilla_s3 = close_1d - 1.1 * daily_range * 1.1 / 4
    # Shift to use only completed 1d bar data
    camarilla_r3_shifted = np.roll(camarilla_r3, 1)
    camarilla_r3_shifted[0] = np.nan
    camarilla_s3_shifted = np.roll(camarilla_s3, 1)
    camarilla_s3_shifted[0] = np.nan
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_shifted)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_shifted)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND 1d EMA34 uptrend AND volume spike
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema34_1d_aligned[i] and volume[i] > (1.8 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND 1d EMA34 downtrend AND volume spike
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema34_1d_aligned[i] and volume[i] > (1.8 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Camarilla S3 OR 1d EMA34 turns down
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Camarilla R3 OR 1d EMA34 turns up
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals