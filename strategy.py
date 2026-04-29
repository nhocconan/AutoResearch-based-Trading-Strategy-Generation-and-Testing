#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Uses Camarilla pivot levels calculated from 1d OHLC to identify key intraday support/resistance
# Breakouts above R3 or below S3 with volume > 2.0x 20-period average trigger entries
# 1d EMA34 filter ensures trades align with higher timeframe trend (long only when price > EMA34, short only when price < EMA34)
# Designed for 6h timeframe to target 50-150 total trades over 4 years (12-37/year) minimizing fee drag
# Works in both bull and bear markets by combining mean-reversion fade at R3/S3 with breakout continuation logic

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
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
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from 1d OHLC (using prior day's close for today's levels)
    # Camarilla: based on previous day's range
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values  # Will shift inside loop to avoid look-ahead
    
    # Calculate ATR(14) for dynamic stop
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 14)  # EMA, volume MA, and ATR warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema = ema_34_aligned[i]
        curr_atr = atr[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Get prior 1d OHLC for Camarilla calculation (shifted by 1 to avoid look-ahead)
        # We need the 1d bar that closed prior to the current 6h bar's time
        # Since 1d data is daily, we use the prior completed 1d bar
        idx_1d = i // 4  # 6h bars per day = 4
        if idx_1d < 1:  # Need at least one prior 1d bar
            signals[i] = 0.0
            continue
            
        # Prior completed 1d bar OHLC (index idx_1d-1)
        phigh = high_1d[idx_1d - 1]
        plow = low_1d[idx_1d - 1]
        pclose = close_1d[idx_1d - 1]
        
        # Calculate Camarilla levels for current day based on prior day's range
        range_ = phigh - plow
        if range_ <= 0:
            signals[i] = 0.0
            continue
            
        # Camarilla levels
        r3 = pclose + range_ * 1.1 / 4
        s3 = pclose - range_ * 1.1 / 4
        r4 = pclose + range_ * 1.1 / 2
        s4 = pclose - range_ * 1.1 / 2
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = curr_volume > 2.0 * curr_vol_ma
        
        # Handle exits and trailing stop
        if position == 1:  # Long position
            # Trailing stop: price drops 2.5*ATR from highest since entry (tracked via close)
            # Simplified: exit if price closes below EMA34 or breaks S3 with volume
            if curr_close < curr_ema or (curr_low < s3 and vol_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Trailing stop: price rises 2.5*ATR from lowest since entry
            # Simplified: exit if price closes above EMA34 or breaks R3 with volume
            if curr_close > curr_ema or (curr_high > r3 and vol_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above R3 with volume confirmation and above 1d EMA34 (uptrend)
            if vol_confirm and curr_high > r3 and curr_close > curr_ema:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 with volume confirmation and below 1d EMA34 (downtrend)
            elif vol_confirm and curr_low < s3 and curr_close < curr_ema:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals