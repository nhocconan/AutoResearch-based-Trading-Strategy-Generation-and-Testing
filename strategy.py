#!/usr/bin/env python3
# 6h_weekly_pivot_volume_confirmation_v2
# Hypothesis: 6h strategy using weekly Camarilla pivot levels (R4/S4 breakout, R3/S3 fade) with volume confirmation (>1.3x 20-bar avg) and HTF 1d EMA(50) trend alignment. 
# Weekly pivot calculated from prior week's OHLC. Breakouts at R4/S4 continue trend; fades at R3/S3 mean-revert. Volume filters low-conviction moves.
# Works in bull/bear: pivot structure provides support/resistance; volume confirms institutional interest; HTF EMA avoids counter-trend trades.
# Target: 12-37 trades/year (50-150 total over 4 years). Discrete size 0.25 to minimize fee churn.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_volume_confirmation_v2"
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
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Multi-timeframe: 1d EMA(50) trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    close_1d_s = pd.Series(close_1d)
    ema_50_1d = close_1d_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Weekly Camarilla pivot levels (using prior week OHLC)
    # Resample to weekly using actual Binance weekly data via mtf_data
    df_1w = get_htf_data(prices, '1w')
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_open = df_1w['open'].values
    
    # Camarilla levels: based on prior week's range
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low)
    # S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    weekly_range = weekly_high - weekly_low
    r4 = weekly_close + 1.5 * weekly_range
    r3 = weekly_close + 1.1 * weekly_range
    s3 = weekly_close - 1.1 * weekly_range
    s4 = weekly_close - 1.5 * weekly_range
    
    # Align weekly levels to 6h timeframe (wait for weekly bar close)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        # HTF trend filter: price above/below 1d EMA(50)
        htf_uptrend = close[i] > ema_50_1d_aligned[i]
        htf_downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below S3 (mean reversion failure) or below S4 (stop)
            if close[i] < s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above R3 (mean reversion failure) or above R4 (stop)
            if close[i] > r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Breakout signals: price breaks R4/S4 with volume and HTF alignment
            bullish_breakout = (close[i] > r4_aligned[i]) and volume_confirmed and htf_uptrend
            bearish_breakout = (close[i] < s4_aligned[i]) and volume_confirmed and htf_downtrend
            
            # Mean reversion signals: price rejects R3/S3 with volume and counter-HTF alignment
            bullish_rejection = (close[i] < r3_aligned[i] and close[i] > r3_aligned[i-1]) and volume_confirmed and htf_downtrend
            bearish_rejection = (close[i] > s3_aligned[i] and close[i] < s3_aligned[i-1]) and volume_confirmed and htf_uptrend
            
            if bullish_breakout or bullish_rejection:
                position = 1
                signals[i] = 0.25
            elif bearish_breakout or bearish_rejection:
                position = -1
                signals[i] = -0.25
    
    return signals