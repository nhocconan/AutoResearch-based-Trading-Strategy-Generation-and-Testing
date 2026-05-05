#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 Breakout with 12h EMA34 Trend Filter and Volume Spike
# Long when price breaks above R3 (camarilla resistance) AND price > 12h EMA34 (uptrend) AND volume spike
# Short when price breaks below S3 (camarilla support) AND price < 12h EMA34 (downtrend) AND volume spike
# Camarilla levels from 12h timeframe provide institutional support/resistance
# 12h EMA34 filters for higher timeframe trend alignment to avoid counter-trend trades
# Volume spike (2.0x 20-bar MA) confirms breakout validity
# Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)
# Timeframe: 6h (primary timeframe as required)
# Target: 50-150 total trades over 4 years (12-37/year) to balance signal quality and fee drag

name = "6h_Camarilla_R3S3_Breakout_12hEMA34_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for Camarilla levels and EMA34
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA34
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate Camarilla levels from previous 12h bar (HLC of completed bar)
    # We need to shift by 1 to use only completed 12h bar data (no look-ahead)
    if len(df_12h) >= 2:
        high_12h = df_12h['high'].values
        low_12h = df_12h['low'].values
        close_12h_shift = df_12h['close'].shift(1).values  # Previous completed bar
        
        # Typical price from previous 12h bar
        typical_price = (high_12h[:-1] + low_12h[:-1] + close_12h_shift[:-1]) / 3.0
        range_12h = high_12h[:-1] - low_12h[:-1]
        
        # Camarilla levels
        R3 = typical_price + (1.1 * range_12h / 2)
        S3 = typical_price - (1.1 * range_12h / 2)
        
        # Align to 6h timeframe (wait for 12h bar to complete)
        R3_aligned = align_htf_to_ltf(prices, df_12h.iloc[:-1], R3, additional_delay_bars=0)
        S3_aligned = align_htf_to_ltf(prices, df_12h.iloc[:-1], S3, additional_delay_bars=0)
        
        # Pad arrays to match length (first value will be NaN due to shift)
        R3_full = np.full(n, np.nan)
        S3_full = np.full(n, np.nan)
        if len(R3_aligned) > 0:
            R3_full[1:len(R3_aligned)+1] = R3_aligned
        if len(S3_aligned) > 0:
            S3_full[1:len(S3_aligned)+1] = S3_aligned
    else:
        R3_full = np.full(n, np.nan)
        S3_full = np.full(n, np.nan)
    
    # Volume confirmation on 6h (threshold: 2.0x)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(R3_full[i]) or np.isnan(S3_full[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 AND price > 12h EMA34 (uptrend) AND volume spike
            if (close[i] > R3_full[i] and 
                close[i] > ema_34_12h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND price < 12h EMA34 (downtrend) AND volume spike
            elif (close[i] < S3_full[i] and 
                  close[i] < ema_34_12h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < 12h EMA34 (trend break) OR price < S3 (support break)
            if close[i] < ema_34_12h_aligned[i] or close[i] < S3_full[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > 12h EMA34 (trend break) OR price > R3 (resistance break)
            if close[i] > ema_34_12h_aligned[i] or close[i] > R3_full[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals