#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d ATR-based volatility breakout with 6h trend filter and volume spike confirmation
# Long when price breaks above upper ATR band (close > upper_band) AND price > 6h EMA50 AND volume > 2.0 * avg_volume(20) on 6h
# Short when price breaks below lower ATR band (close < lower_band) AND price < 6h EMA50 AND volume > 2.0 * avg_volume(20) on 6h
# Exit when price crosses back through 6h EMA50 OR volume drops below average
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# ATR bands capture volatility expansion breakouts that work in both bull and bear markets
# 6h EMA50 filters for primary trend alignment to avoid counter-trend trades
# Volume spike confirms breakout strength and reduces false signals
# Works in bull markets (buying breakouts in uptrend) and bear markets (selling breakdowns in downtrend)

name = "6h_ATRBreakout_6hEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:  # Need at least 14 daily bars for ATR14
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR(14) on daily timeframe
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First TR is undefined
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate ATR bands: upper = close + 2*ATR, lower = close - 2*ATR
    upper_band_1d = close_1d + (2.0 * atr_14)
    lower_band_1d = close_1d - (2.0 * atr_14)
    
    # Align ATR bands to 6h timeframe (wait for completed daily bar)
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band_1d)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band_1d)
    
    # Get 6h data ONCE before loop for EMA50 trend filter
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:  # Need enough for EMA50
        return np.zeros(n)
    close_6h = df_6h['close'].values
    
    # Calculate 6h EMA50
    close_6h_series = pd.Series(close_6h)
    ema50_6h = close_6h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_6h_aligned = align_htf_to_ltf(prices, df_6h, ema50_6h)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or 
            np.isnan(ema50_6h_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper ATR band, above 6h EMA50, volume confirmation, in session
            if (close[i] > upper_band_aligned[i] and 
                close[i] > ema50_6h_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower ATR band, below 6h EMA50, volume confirmation, in session
            elif (close[i] < lower_band_aligned[i] and 
                  close[i] < ema50_6h_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below 6h EMA50 OR volume drops below average
            if close[i] < ema50_6h_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above 6h EMA50 OR volume drops below average
            if close[i] > ema50_6h_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals