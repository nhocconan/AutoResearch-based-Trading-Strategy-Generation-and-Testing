#!/usr/bin/env python3
"""
1d_Camarilla_R3_S3_Breakout_1wEMA50_Trend_VolumeSpike_v1
Hypothesis: Trade daily Camarilla R3/S3 breakouts with 1-week EMA50 trend filter and volume spike confirmation.
- Trend filter: price > 1w close + 0.5*ATR(14) = bullish, price < 1w close - 0.5*ATR(14) = bearish, else ranging.
- In trending markets: buy breakouts above R3, sell breakdowns below S3.
- In ranging markets: fade extremes at R3/S3 with mean reversion to pivot.
- Volume confirmation: require volume > 1.5x 20-period average to avoid false breakouts.
- Position size: 0.25. Target: 30-100 total trades over 4 years = 7-25/year.
- Works in both bull and bear: ATR trend filter adapts to volatility regime, volume filters noise.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate 1w ATR(14) for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range calculation
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # ATR(14) using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    atr_14_1w = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    # Calculate daily Camarilla pivot levels (using previous day's OHLC)
    # Need to resample daily OHLC from 15m data? No - we can use 1d data from prices if available
    # Since we're on 1d timeframe, we can use the daily OHLC directly
    # But we need previous day's OHLC for today's Camarilla levels
    # For 1d timeframe, we can shift the daily OHLC by 1
    
    # Since prices is already 1d timeframe, we can use:
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_ = prev_high - prev_low
    
    # Camarilla levels
    r1 = pivot + (range_ * 1.1 / 12)
    s1 = pivot - (range_ * 1.1 / 12)
    r3 = pivot + (range_ * 1.1 / 4)
    s3 = pivot - (range_ * 1.1 / 4)
    
    # Volume spike confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for ATR(14) and volume MA (20)
    start_idx = max(14, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr_14_1w_aligned[i]) or 
            np.isnan(r3[i]) or
            np.isnan(s3[i]) or
            np.isnan(pivot[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1w HTF trend using ATR bands
        # Align 1w close to 1d timeframe
        close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
        htf_1w_bullish = close[i] > (close_1w_aligned[i] + (0.5 * atr_14_1w_aligned[i]))
        htf_1w_bearish = close[i] < (close_1w_aligned[i] - (0.5 * atr_14_1w_aligned[i]))
        
        # Determine if we are in trending or ranging market based on ATR bands
        trending_market = htf_1w_bullish or htf_1w_bearish
        ranging_market = not trending_market
        
        if position == 0:
            if trending_market:
                # Trending market: trade breakout continuation
                long_setup = (close[i] > r3[i]) and htf_1w_bullish and volume_spike[i]
                short_setup = (close[i] < s3[i]) and htf_1w_bearish and volume_spike[i]
            else:
                # Ranging market: trade mean reversion at extremes
                long_setup = (close[i] < s3[i]) and (close[i] > s1[i]) and volume_spike[i]  # Oversold bounce
                short_setup = (close[i] > r3[i]) and (close[i] < r1[i]) and volume_spike[i]  # Overbought rejection
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions
            if trending_market:
                # In trending market: exit on trend reversal or touch of S3
                exit_signal = (not htf_1w_bullish) or (close[i] < s3[i])
            else:
                # In ranging market: exit on mean reversion to pivot or touch of R3
                exit_signal = (close[i] > pivot[i]) or (close[i] > r3[i])
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions
            if trending_market:
                # In trending market: exit on trend reversal or touch of R3
                exit_signal = htf_1w_bullish or (close[i] > r3[i])
            else:
                # In ranging market: exit on mean reversion to pivot or touch of S3
                exit_signal = (close[i] < pivot[i]) or (close[i] < s3[i])
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R3_S3_Breakout_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0