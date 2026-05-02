#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot (R3/S3) breakout with 1w EMA34 trend filter and volume confirmation
# Uses 12h timeframe for signal generation with Camarilla R3/S3 breakouts
# 1w EMA34 provides multi-timeframe trend filter to avoid counter-trend trades
# Volume confirmation (1.5x 20-period average) ensures institutional participation
# Chop regime filter from 12h timeframe avoids ranging markets (CHOP > 61.8 = range)
# Discrete position sizing (0.25) minimizes fee churn
# Target: 50-150 total trades over 4 years = 12-37/year for 12h timeframe
# Works in bull markets via trend-aligned breakouts, in bear via chop filter avoiding false signals
# Designed for low trade frequency to minimize fee drag (critical for 12h timeframe)

name = "12h_Camarilla_R3_S3_Breakout_1wEMA34_Trend_VolumeSpike_v1"
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
    open_price = prices['open'].values  # For Camarilla calculation
    
    # Load 1w data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Camarilla levels (R3, S3) from previous 12h bar
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    # Using previous bar's OHLC to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_open = np.roll(open_price, 1)
    
    # Set first bar's previous values to NaN (no previous bar)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    prev_open[0] = np.nan
    
    # Calculate Camarilla R3 and S3
    camarilla_range = prev_high - prev_low
    r3 = prev_close + 1.1 * camarilla_range / 2
    s3 = prev_close - 1.1 * camarilla_range / 2
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.5)
    
    # Calculate 12h Chopiness Index (14) - trending when < 38.2, ranging when > 61.8
    # True Range
    tr1 = np.abs(high[1:] - low[:-1])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR14
    atr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Chop = 100 * log15(sum(ATR14)/ (max(high)-min(low)) over 14 periods)
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log15(atr14 * 14 / (max_high - min_low))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(volume_confirm[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when Chop < 61.8 (not strongly ranging)
        if chop[i] > 61.8:
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Camarilla R3 + price > 1w EMA34 + volume confirm
            if close[i] > r3[i] and close[i] > ema_34_1w_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S3 + price < 1w EMA34 + volume confirm
            elif close[i] < s3[i] and close[i] < ema_34_1w_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Camarilla S3 or reverse signal
            if close[i] < s3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Camarilla R3 or reverse signal
            if close[i] > r3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals