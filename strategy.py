#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian channel breakout with volume confirmation and ATR filter.
# Long when price breaks above 1d Donchian(20) high AND volume > 1.5x 20-period average volume AND ATR(14) < 0.05 * close (low volatility).
# Short when price breaks below 1d Donchian(20) low AND volume > 1.5x 20-period average volume AND ATR(14) < 0.05 * close.
# Exit when price crosses the 1d Donchian midpoint (mean of high and low channel).
# Uses discrete position size 0.25. Donchian breakouts capture sustained moves in both bull and bear markets.
# Volume confirmation ensures breakouts are supported by participation, reducing false signals.
# ATR filter avoids high-volatility choppy markets where breakouts often fail.
# 12h timeframe targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Works in bull markets (catch uptrend breakouts) and bear markets (catch downtrend breakdowns).

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Donchian channel, volume average, and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 1d Indicators: Donchian Channel (20-period) ===
    # Upper band = max(high, lookback=20)
    # Lower band = min(low, lookback=20)
    # Middle band = (upper + lower) / 2
    lookback = 20
    upper_1d = pd.Series(high_1d).rolling(window=lookback, min_periods=lookback).max().values
    lower_1d = pd.Series(low_1d).rolling(window=lookback, min_periods=lookback).min().values
    middle_1d = (upper_1d + lower_1d) / 2.0
    
    # === 1d Indicators: Volume Average (20-period) ===
    vol_ma_1d = pd.Series(volume_1d).rolling(window=lookback, min_periods=lookback).mean().values
    
    # === 1d Indicators: ATR (14-period) ===
    # True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
    prev_close_1d = np.roll(close_1d, 1)
    prev_close_1d[0] = np.nan
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - prev_close_1d)
    tr3 = np.abs(low_1d - prev_close_1d)
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Align all indicators to primary timeframe (12h)
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    middle_aligned = align_htf_to_ltf(prices, df_1d, middle_1d)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 30  # Donchian(20) and ATR(14) need sufficient warmup
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(middle_aligned[i]) or np.isnan(vol_ma_aligned[i]) or 
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        upper = upper_aligned[i]
        lower = lower_aligned[i]
        middle = middle_aligned[i]
        vol_ma = vol_ma_aligned[i]
        atr = atr_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Volatility filter: avoid high volatility (ATR > 5% of price)
        vol_filter = atr < 0.05 * price
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = vol > 1.5 * vol_ma
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price < middle (break below Donchian midpoint)
            if price < middle:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price > middle (break above Donchian midpoint)
            if price > middle:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: price > upper (break above Donchian high) AND volume confirmation AND low volatility
            if (price > upper) and vol_confirm and vol_filter:
                signals[i] = 0.25
                position = 1
            
            # SHORT: price < lower (break below Donchian low) AND volume confirmation AND low volatility
            elif (price < lower) and vol_confirm and vol_filter:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "12h_1dDonchian20_VolumeConfirmation_ATRFilter_V1"
timeframe = "12h"
leverage = 1.0