#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout/retest with 1w trend filter and volume confirmation
# - Camarilla levels (R3,R4,S3,S4) from 1d: R3/S3 = mean reversion zones, R4/S4 = breakout zones
# - Long when price retraces to S3 in uptrend (1w close > 1w open) with volume > 1.5x 20-period average
# - Short when price retraces to R3 in downtrend (1w close < 1w open) with volume > 1.5x 20-period average
# - Breakout continuation: Long on close > R4 in uptrend, Short on close < S4 in downtrend
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits for 6h
# - Works in bull (retests/breakouts in uptrend) and bear (retests/breakdowns in downtrend)

name = "6h_1d_1w_camarilla_retest_v1"
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return signals
    
    # Pre-compute 1d Camarilla levels (using previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R4 = close + 1.1*(high-low)*1.1/2
    # R3 = close + 1.1*(high-low)*1.1/4
    # S3 = close - 1.1*(high-low)*1.1/4
    # S4 = close - 1.1*(high-low)*1.1/2
    rng = high_1d - low_1d
    camarilla_r4 = close_1d + 1.1 * rng * 1.1 / 2
    camarilla_r3 = close_1d + 1.1 * rng * 1.1 / 4
    camarilla_s3 = close_1d - 1.1 * rng * 1.1 / 4
    camarilla_s4 = close_1d - 1.1 * rng * 1.1 / 2
    
    # Pre-compute 1w trend: 1 if bullish (close > open), -1 if bearish (close < open)
    open_1w = df_1w['open'].values
    close_1w = df_1w['close'].values
    weekly_trend = np.where(close_1w > open_1w, 1, -1)  # 1=bullish, -1=bearish
    
    # Pre-compute 1d volume SMA (20-period)
    volume_1d = df_1d['volume'].values
    volume_series = pd.Series(volume_1d)
    volume_sma_20_1d = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Align 1w trend to 6h timeframe (no extra delay needed for trend direction)
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(volume_sma_20_aligned[i]) or np.isnan(weekly_trend_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20_aligned[i]
        
        # Weekly trend filter
        trend_bullish = weekly_trend_aligned[i] == 1
        trend_bearish = weekly_trend_aligned[i] == -1
        
        # Camarilla levels
        r3 = camarilla_r3_aligned[i]
        r4 = camarilla_r4_aligned[i]
        s3 = camarilla_s3_aligned[i]
        s4 = camarilla_s4_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long conditions:
        # 1. Retest to S3 in uptrend (mean reversion)
        # 2. Breakout above R4 in uptrend (continuation)
        if trend_bullish and vol_confirm:
            # Mean reversion long: price touches or goes below S3 then closes back above it
            if price_low <= s3 and price_close > s3:
                enter_long = True
            # Breakout long: price closes above R4
            elif price_close > r4:
                enter_long = True
        
        # Short conditions:
        # 1. Retest to R3 in downtrend (mean reversion)
        # 2. Breakdown below S4 in downtrend (continuation)
        if trend_bearish and vol_confirm:
            # Mean reversion short: price touches or goes above R3 then closes back below it
            if price_high >= r3 and price_close < r3:
                enter_short = True
            # Breakdown short: price closes below S4
            elif price_close < s4:
                enter_short = True
        
        # Exit conditions: opposite retest or trend change
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price retests R3 (mean reversion failure) or trend turns bearish
            exit_long = (price_high >= r3 and price_close < r3) or (not trend_bullish)
        elif position == -1:
            # Exit short if price retests S3 (mean reversion failure) or trend turns bullish
            exit_short = (price_low <= s3 and price_close > s3) or (not trend_bearish)
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals