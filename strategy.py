#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 4h Supertrend(ATR=10,mult=3) for trend direction,
# 4h RSI(14) for mean-reversion entries within the trend, and 1d volume spike filter.
# Long when Supertrend is bullish, RSI < 30 (oversold), and 1d volume > 2.0x 20-period median volume.
# Short when Supertrend is bearish, RSI > 70 (overbought), and same volume condition.
# Exit via ATR(10) trailing stop: long exits when price < highest high since entry - 3.0*ATR,
# short exits when price > lowest low since entry + 3.0*ATR.
# Uses discrete position size 0.25. Target: 50-150 total trades over 4 years (12-37/year).
# Supertrend captures the trend, RSI provides mean-reversion entries in the trend direction,
# volume spike confirms institutional interest, wider ATR stop reduces whipsaws.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h data once before loop for Supertrend, RSI, and ATR
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # === 4h Indicators: Supertrend, RSI(14), ATR(10) ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # ATR(10)
    tr1 = pd.Series(high_4h - low_4h)
    tr2 = pd.Series(np.abs(high_4h - np.roll(close_4h, 1)))
    tr3 = pd.Series(np.abs(low_4h - np.roll(close_4h, 1)))
    tr2.iloc[0] = tr1.iloc[0]
    tr3.iloc[0] = tr1.iloc[0]
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_10 = tr.rolling(window=10, min_periods=10).mean().values
    
    # Supertrend(ATR=10, mult=3)
    hl2 = (high_4h + low_4h) / 2
    upper_band = hl2 + (3.0 * atr_10)
    lower_band = hl2 - (3.0 * atr_10)
    
    upper_band = pd.Series(upper_band)
    lower_band = pd.Series(lower_band)
    
    for i in range(1, len(upper_band)):
        if close_4h[i-1] <= upper_band[i-1]:
            upper_band[i] = min(upper_band[i], upper_band[i-1])
        else:
            upper_band[i] = upper_band[i]
        
        if close_4h[i-1] >= lower_band[i-1]:
            lower_band[i] = max(lower_band[i], lower_band[i-1])
        else:
            lower_band[i] = lower_band[i]
    
    upper_band = upper_band.values
    lower_band = lower_band.values
    
    supertrend = np.zeros_like(close_4h)
    supertrend[:] = np.nan
    dir_ = np.ones_like(close_4h)  # 1 for uptrend, -1 for downtrend
    
    for i in range(10, len(supertrend)):
        if close_4h[i] > upper_band[i-1]:
            dir_[i] = 1
        elif close_4h[i] < lower_band[i-1]:
            dir_[i] = -1
        else:
            dir_[i] = dir_[i-1]
        
        if dir_[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    # RSI(14)
    delta = pd.Series(close_4h).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14 = rsi_14.values
    
    # Get 1d data for volume median
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_median_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).median().values
    
    # Align all indicators to primary timeframe (4h)
    supertrend_aligned = align_htf_to_ltf(prices, df_4h, supertrend)
    dir_aligned = align_htf_to_ltf(prices, df_4h, dir_)
    rsi_aligned = align_htf_to_ltf(prices, df_4h, rsi_14)
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr_10)
    vol_median_aligned = align_htf_to_ltf(prices, df_1d, vol_median_20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(14, 10, 14, 10, 20)  # RSI(14), Supertrend, ATR(10), volume median(20)
    
    # Track position state for trailing stops
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(supertrend_aligned[i]) or np.isnan(dir_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or np.isnan(atr_aligned[i]) or np.isnan(vol_median_aligned[i])):
            signals[i] = 0.0
            position = 0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
            continue
        
        # Current values (aligned)
        st = supertrend_aligned[i]
        direction = dir_aligned[i]
        rsi = rsi_aligned[i]
        atr = atr_aligned[i]
        vol_median = vol_median_aligned[i]
        price = close[i]
        
        # Get current 1d volume for volume spike filter
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        current_vol_1d = vol_1d_aligned[i]
        
        # Volume spike filter: current 1d volume > 2.0x median volume
        volume_spike = current_vol_1d > (vol_median * 2.0)
        
        # Trend and mean-reversion filters
        uptrend = direction == 1
        downtrend = direction == -1
        oversold = rsi < 30
        overbought = rsi > 70
        
        # === EXIT LOGIC (trailing stop) ===
        exit_signal = False
        if position == 1:  # long position
            # Update highest high since entry
            if price > highest_since_entry:
                highest_since_entry = price
            # Exit when price drops below highest high - 3.0*ATR
            if price < highest_since_entry - 3.0 * atr:
                exit_signal = True
        elif position == -1:  # short position
            # Update lowest low since entry
            if price < lowest_since_entry:
                lowest_since_entry = price
            # Exit when price rises above lowest low + 3.0*ATR
            if price > lowest_since_entry + 3.0 * atr:
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
            # Uptrend, RSI oversold, and volume spike
            if uptrend and oversold and volume_spike:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price  # initialize trailing stop
            
            # SHORT CONDITIONS
            # Downtrend, RSI overbought, and volume spike
            elif downtrend and overbought and volume_spike:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price  # initialize trailing stop
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_Supertrend10_3_RSI14_OBOS_VolumeSpike2.0x_ATRTrail3.0_v1"
timeframe = "4h"
leverage = 1.0