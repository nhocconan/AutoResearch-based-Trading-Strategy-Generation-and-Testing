#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian(20) breakout for trend direction,
# 12h RSI(14) for mean-reversion entries within the trend, and 1d volume spike filter.
# Long when price breaks above 1d Donchian upper channel, RSI < 30 (oversold),
# and 1d volume > 1.5x 20-period median volume. Short when price breaks below
# 1d Donchian lower channel, RSI > 70 (overbought), and same volume condition.
# Exit via ATR(10) trailing stop: long exits when price < highest high since entry - 2.5*ATR,
# short exits when price > lowest low since entry + 2.5*ATR.
# Uses discrete position size 0.25. Target: 50-150 total trades over 4 years (12-37/year).
# Donchian captures the trend structure, RSI provides mean-reversion entries in the trend direction,
# volume spike confirms institutional interest, moderate ATR stop reduces whipsaws.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data once before loop for Donchian, RSI, ATR, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicators: Donchian(20), RSI(14), ATR(10), Volume Median ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # ATR(10)
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr2.iloc[0] = tr1.iloc[0]
    tr3.iloc[0] = tr1.iloc[0]
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_10 = tr.rolling(window=10, min_periods=10).mean().values
    
    # Donchian(20)
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # RSI(14)
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14 = rsi_14.values
    
    # Volume median(20)
    vol_median_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).median().values
    
    # Get 12h data for price (to check breakout)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Align all indicators to primary timeframe (12h)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_10)
    vol_median_aligned = align_htf_to_ltf(prices, df_1d, vol_median_20)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    close_12h = df_12h['close'].values
    close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 14, 10, 20)  # Donchian(20), RSI(14), ATR(10), volume median(20)
    
    # Track position state for trailing stops
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or np.isnan(atr_aligned[i]) or np.isnan(vol_median_aligned[i]) or
            np.isnan(volume_1d_aligned[i]) or np.isnan(close_12h_aligned[i])):
            signals[i] = 0.0
            position = 0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
            continue
        
        # Current values (aligned)
        price_12h = close_12h_aligned[i]
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        rsi = rsi_aligned[i]
        atr = atr_aligned[i]
        vol_median = vol_median_aligned[i]
        current_vol_1d = volume_1d_aligned[i]
        
        # Volume spike filter: current 1d volume > 1.5x median volume
        volume_spike = current_vol_1d > (vol_median * 1.5)
        
        # Breakout and mean-reversion filters
        breakout_up = price_12h > upper
        breakout_down = price_12h < lower
        oversold = rsi < 30
        overbought = rsi > 70
        
        # === EXIT LOGIC (trailing stop) ===
        exit_signal = False
        if position == 1:  # long position
            # Update highest high since entry
            if price_12h > highest_since_entry:
                highest_since_entry = price_12h
            # Exit when price drops below highest high - 2.5*ATR
            if price_12h < highest_since_entry - 2.5 * atr:
                exit_signal = True
        elif position == -1:  # short position
            # Update lowest low since entry
            if price_12h < lowest_since_entry:
                lowest_since_entry = price_12h
            # Exit when price rises above lowest low + 2.5*ATR
            if price_12h > lowest_since_entry + 2.5 * atr:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG CONDITIONS
            # Breakout above upper channel, RSI oversold, and volume spike
            if breakout_up and oversold and volume_spike:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price_12h  # initialize trailing stop
            
            # SHORT CONDITIONS
            # Breakout below lower channel, RSI overbought, and volume spike
            elif breakout_down and overbought and volume_spike:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price_12h  # initialize trailing stop
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "12h_Donchian20_1dRSI14_OBOS_VolumeSpike1.5x_ATRTrail2.5_v1"
timeframe = "12h"
leverage = 1.0