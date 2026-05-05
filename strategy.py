#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Long when: price breaks above 6h Camarilla R3 AND 1d EMA34 is rising (close > EMA34) AND volume > 1.5x 20-period MA
# Short when: price breaks below 6h Camarilla S3 AND 1d EMA34 is falling (close < EMA34) AND volume > 1.5x 20-period MA
# Exit when: price returns to 6h Camarilla pivot point (PP) OR opposite breakout occurs
# Uses Camarilla for precise intraday levels, 1d EMA for higher-timeframe trend, volume for conviction
# Timeframe: 6h, HTF: 1d. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate volume confirmation on 6h using 20-period MA
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Calculate 6h Camarilla levels (based on prior 6h bar's OHLC)
    if len(high) >= 2 and len(low) >= 2 and len(close) >= 2:
        # Use prior bar's OHLC to avoid look-ahead
        prev_high = np.roll(high, 1)
        prev_low = np.roll(low, 1)
        prev_close = np.roll(close, 1)
        prev_high[0] = np.nan  # first bar has no prior
        prev_low[0] = np.nan
        prev_close[0] = np.nan
        
        # Camarilla calculation: PP = (H+L+C)/3
        camarilla_pp = (prev_high + prev_low + prev_close) / 3.0
        camarilla_range = prev_high - prev_low
        camarilla_r3 = camarilla_pp + (camarilla_range * 1.1 / 4)
        camarilla_s3 = camarilla_pp - (camarilla_range * 1.1 / 4)
    else:
        camarilla_pp = np.full(n, np.nan)
        camarilla_r3 = np.full(n, np.nan)
        camarilla_s3 = np.full(n, np.nan)
    
    # Camarilla breakout signals
    camarilla_breakout_up = (close > camarilla_r3) & (np.roll(close, 1) <= np.roll(camarilla_r3, 1))
    camarilla_breakout_down = (close < camarilla_s3) & (np.roll(close, 1) >= np.roll(camarilla_s3, 1))
    camarilla_revert_pp = np.abs(close - camarilla_pp) < (0.001 * close)  # approximate PP return
    
    # Get 1d data ONCE before loop for EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # need sufficient data for EMA34
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # 1d trend: rising when close > EMA34, falling when close < EMA34
    ema34_bullish = close_1d > ema_34_1d
    ema34_bearish = close_1d < ema_34_1d
    
    # Align 1d EMA34 trend to 6h timeframe
    ema34_bullish_aligned = align_htf_to_ltf(prices, df_1d, ema34_bullish.astype(float))
    ema34_bearish_aligned = align_htf_to_ltf(prices, df_1d, ema34_bearish.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(camarilla_pp[i]) or 
            np.isnan(ema34_bullish_aligned[i]) or np.isnan(ema34_bearish_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Camarilla breakout up + 1d EMA34 bullish + volume filter
            if (camarilla_breakout_up[i] and 
                ema34_bullish_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Camarilla breakout down + 1d EMA34 bearish + volume filter
            elif (camarilla_breakout_down[i] and 
                  ema34_bearish_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Camarilla PP OR short breakout occurs
            if (camarilla_revert_pp[i] or camarilla_breakout_down[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Camarilla PP OR long breakout occurs
            if (camarilla_revert_pp[i] or camarilla_breakout_up[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals