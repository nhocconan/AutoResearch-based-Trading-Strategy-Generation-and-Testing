#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Uses 12h timeframe for signal generation with Camarilla R3/S3 breakout levels
# 1d EMA34 provides higher timeframe trend filter to avoid counter-trend trades
# Volume confirmation (1.8x 20-period average) ensures institutional participation
# Chop regime filter from 1d timeframe avoids ranging markets (CHOP > 61.8 = range)
# Discrete position sizing (0.28) balances return and drawdown
# Target: 80-120 total trades over 4 years = 20-30/year for 12h timeframe
# Works in bull markets via trend-aligned breakouts, in bear via chop filter avoiding false signals
# Designed for low trade frequency to minimize fee drag (critical for 12h timeframe)

name = "12h_Camarilla_R3S3_1dEMA34_VolumeS_ChopFilter_v1"
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
    
    # Pre-compute session hours (08-20 UTC) - index is DatetimeIndex
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop for EMA trend filter and Chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Chopiness Index (14) - trending when < 38.2, ranging when > 61.8
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR14
    atr1 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Chop = 100 * log15(sum(ATR14)/ (max(high)-min(low)) over 14 periods)
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log15(atr1 * 14 / (max_high - min_low))
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate Camarilla levels (R3, S3) on 12h timeframe from prior day's OHLC
    # Camarilla levels: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4
    #                 S3 = close - 1.1*(high-low)*1.1/4, S4 = close - 1.1*(high-low)*1.1/2
    # We use prior 1d OHLC to calculate today's 12h Camarilla levels
    
    # Resample 12h close to daily OHLC for Camarilla calculation
    # Since we cannot resample, we use the 1d data directly for Camarilla levels
    # Camarilla levels are based on prior day's range applied to current price
    
    # Calculate prior day's range (using 1d data shifted by 1)
    prior_high = np.roll(high_1d, 1)
    prior_low = np.roll(low_1d, 1)
    prior_close = np.roll(close_1d, 1)
    prior_high[0] = np.nan  # First value has no prior
    prior_low[0] = np.nan
    prior_close[0] = np.nan
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_range = prior_high - prior_low
    r3_level = prior_close + 1.1 * camarilla_range * 1.1 / 4
    s3_level = prior_close - 1.1 * camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_level)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_level)
    
    # Volume confirmation (1.8x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when Chop < 61.8 (not strongly ranging)
        if chop_aligned[i] > 61.8:
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Camarilla R3 + price > 1d EMA34 + volume confirm
            if close[i] > r3_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.28
                position = 1
            # Short: Price breaks below Camarilla S3 + price < 1d EMA34 + volume confirm
            elif close[i] < s3_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.28
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Camarilla S3 or reverse signal
            if close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Camarilla R3 or reverse signal
            if close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals