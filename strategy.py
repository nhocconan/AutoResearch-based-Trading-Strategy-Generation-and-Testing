#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_pullback_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily OHLC for Camarilla pivot levels (previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot calculation (previous day)
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Resistance and support levels (previous day's data)
    r3 = close_1d + range_1d * 1.166
    s3 = close_1d - range_1d * 1.166
    
    # Shift by 1 to use only completed daily bars (previous day's levels)
    r3 = np.roll(r3, 1)
    s3 = np.roll(s3, 1)
    r3[0] = np.nan
    s3[0] = np.nan
    
    # Align daily Camarilla levels to 12h timeframe
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    pivot_12h = align_htf_to_ltf(prices, df_1d, pivot)
    
    # 12h ATR for volatility filter (14 period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 12h volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or
            np.isnan(pivot_12h[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation (1.5x average)
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Pullback entry: price pulls back to S3/R3 after breaking out
        # Long: price touches S3 from above after being below it (bounce)
        # Short: price touches R3 from below after being above it (rejection)
        
        # Need previous bar's position relative to levels
        if i > 0:
            prev_close = close[i-1]
            prev_r3 = r3_12h[i-1]
            prev_s3 = s3_12h[i-1]
            
            # Long: price was below S3, now touches or goes above S3 (bounce off support)
            long_signal = volume_confirmed and (prev_close <= prev_s3) and (price_low <= s3_12h[i]) and (price_close > s3_12h[i])
            
            # Short: price was above R3, now touches or goes below R3 (rejection at resistance)
            short_signal = volume_confirmed and (prev_close >= prev_r3) and (price_high >= r3_12h[i]) and (price_close < r3_12h[i])
        else:
            long_signal = False
            short_signal = False
        
        # Exit when price returns to the daily pivot (mean reversion)
        exit_long = position == 1 and price_close < pivot_12h[i]
        exit_short = position == -1 and price_close > pivot_12h[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Daily Camarilla pullback strategy for 12h timeframe.
# Enters long when price pulls back to and bounces off daily S3 level (close - 1.166*range) with volume confirmation.
# Enters short when price pulls back to and gets rejected at daily R3 level (close + 1.166*range) with volume confirmation.
# Exits when price returns to the daily pivot level (mean reversion within the day's range).
# Uses pullback entries rather than breakouts to avoid false breakouts and improve win rate.
# Volume confirmation (>1.5x average) ensures institutional participation.
# Targets 15-25 trades per year to stay well within optimal range while maintaining edge.
# Position size: 0.25 to balance risk and return.
# Works in both bull and bear markets as it fades intraday extremes toward the daily pivot.
# Timeframe: 12h (chosen per experiment requirements). HTF: 1d.