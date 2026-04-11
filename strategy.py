#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout/mean reversion + 12h ADX regime filter + volume confirmation
# - Camarilla levels from 12h: R3/S3 for mean reversion (fade), R4/S4 for breakout continuation
# - Trend regime: 12h ADX > 25 = trending (breakout), ADX < 20 = ranging (mean revert)
# - Volume confirmation: 6h volume > 1.3x 20-period average
# - Discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Camarilla pivots work in both bull (breakouts at R4/S4) and bear (mean reversion at R3/S3) markets
# - ADX regime filter ensures we use the correct logic for market conditions
# - Volume confirmation filters out false breakouts/breakdowns

name = "6h_12h_camarilla_adx_volume_v1"
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
    
    # Load 12h data ONCE before loop for Camarilla and ADX
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return signals
    
    # Pre-compute 12h OHLC for Camarilla
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Camarilla levels (based on previous day's range)
    # Camarilla: R4 = close + 1.1*(high-low)/2, R3 = close + 1.1*(high-low)/4
    #          S3 = close - 1.1*(high-low)/4, S4 = close - 1.1*(high-low)/2
    hl_range_12h = high_12h - low_12h
    camarilla_r4 = close_12h + 1.1 * hl_range_12h / 2
    camarilla_r3 = close_12h + 1.1 * hl_range_12h / 4
    camarilla_s3 = close_12h - 1.1 * hl_range_12h / 4
    camarilla_s4 = close_12h - 1.1 * hl_range_12h / 2
    
    # Align Camarilla levels to 6h timeframe (completed 12h bar only)
    r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    
    # Pre-compute 12h ADX for regime filter
    # ADX calculation: +DM, -DM, TR, then smoothed
    plus_dm = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    minus_dm = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    tr = np.maximum(high_12h[1:] - low_12h[1:], 
                    np.maximum(np.abs(high_12h[1:] - close_12h[:-1]), 
                               np.abs(low_12h[1:] - close_12h[:-1])))
    
    # Pad first element
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    # Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = atr[i-1] * (1 - alpha) + tr[i] * alpha
    
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=alpha, adjust=False).mean().values
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Pre-compute 6h volume confirmation (20-period average)
    volume_series = pd.Series(volume)
    volume_sma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Regime filter from 12h ADX
        trending = adx_aligned[i] > 25
        ranging = adx_aligned[i] < 20
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume_current > 1.3 * volume_sma_20[i]
        
        # Entry conditions based on regime
        enter_long = False
        enter_short = False
        
        if trending:
            # Trending market: breakout continuation at R4/S4
            if price_close > r4_aligned[i] and vol_confirm:
                enter_long = True
            if price_close < s4_aligned[i] and vol_confirm:
                enter_short = True
        elif ranging:
            # Ranging market: mean reversion at R3/S3
            if price_close < r3_aligned[i] and price_close > s3_aligned[i]:
                # In the range, look for reversals from extremes
                if price_close <= s3_aligned[i] * 1.002:  # Near S3, go long
                    enter_long = True
                if price_close >= r3_aligned[i] * 0.998:  # Near R3, go short
                    enter_short = True
        
        # Exit conditions: opposite signal or regime change
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price reaches R3 in ranging or breaks S4 in trending
            if ranging and price_close >= r3_aligned[i]:
                exit_long = True
            if trending and price_close < s4_aligned[i]:
                exit_long = True
        elif position == -1:
            # Exit short if price reaches S3 in ranging or breaks R4 in trending
            if ranging and price_close <= s3_aligned[i]:
                exit_short = True
            if trending and price_close > r4_aligned[i]:
                exit_short = True
        
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