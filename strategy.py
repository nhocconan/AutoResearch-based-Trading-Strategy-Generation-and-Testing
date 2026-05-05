#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 Breakout with 1d EMA34 Trend and Volume Spike
# Long when price breaks above Camarilla R3 level AND price > 1d EMA34 (uptrend) AND volume spike
# Short when price breaks below Camarilla S3 level AND price < 1d EMA34 (downtrend) AND volume spike
# Camarilla levels provide intraday support/resistance; EMA34 filters HTF trend; volume confirms institutional interest
# Timeframe: 4h (primary timeframe as required)
# Target: 75-200 total trades over 4 years (19-50/year) to balance signal quality and fee drag

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike"
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
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels for each 4h bar using prior 4h bar's OHLC
    # Need prior bar's OHLC, so we use shift(1) concept via rolling window
    camarilla_R3 = np.full(n, np.nan)
    camarilla_S3 = np.full(n, np.nan)
    
    if len(close) >= 2:
        # For each bar i, use OHLC from bar i-1 (prior completed bar)
        # Since we're in a loop, we'll calculate these inside the loop for simplicity
        pass  # Will calculate inside loop
    
    # Volume confirmation on 4h (threshold: 2.0x)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(2, n):  # Start from 2 to ensure we have prior bar for Camarilla
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels using prior completed bar (i-1)
        prior_high = high[i-1]
        prior_low = low[i-1]
        prior_close = close[i-1]
        
        # Camarilla formula
        rng = prior_high - prior_low
        camarilla_R3 = prior_close + rng * 1.1 / 4
        camarilla_S3 = prior_close - rng * 1.1 / 4
        
        if position == 0:
            # Long: price breaks above Camarilla R3 AND price > 1d EMA34 (uptrend) AND volume spike
            if (close[i] > camarilla_R3 and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 AND price < 1d EMA34 (downtrend) AND volume spike
            elif (close[i] < camarilla_S3 and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below camarilla S3 OR price < 1d EMA34 (trend break)
            if close[i] < camarilla_S3 or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above camarilla R3 OR price > 1d EMA34 (trend break)
            if close[i] > camarilla_R3 or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals