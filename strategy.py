#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze Breakout + 1w Camarilla Pivots + Volume Confirmation
# Bollinger Band Squeeze: Identifies low volatility periods (BB Width < 20th percentile) preceding breakouts
# Breakout direction determined by 1w Camarilla pivot levels: Long above R3, Short below S3
# Volume confirmation ensures participation (>1.5x 20-period average)
# Works in both bull and bear markets by capturing volatility expansion after consolidation
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing (0.25) to manage fee drag
# 6h timeframe balances responsiveness with reasonable trade frequency

name = "6h_BB_Squeeze_1wCamarilla_Breakout_Volume"
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
    
    # Get 1w data ONCE before loop for Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate 1w Camarilla pivot levels (using previous week's OHLC)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low)
    #          S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    prev_week_close = df_1w['close'].shift(1).values
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    
    camarilla_r3 = prev_week_close + 1.1 * (prev_week_high - prev_week_low)
    camarilla_s3 = prev_week_close - 1.1 * (prev_week_high - prev_week_low)
    
    # Align 1w Camarilla levels to 6h timeframe (wait for weekly close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3, additional_delay_bars=1)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3, additional_delay_bars=1)
    
    # Bollinger Bands on 6h (20, 2)
    if len(close) >= 20:
        ma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
        std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
        upper_bb = ma_20 + 2 * std_20
        lower_bb = ma_20 - 2 * std_20
        bb_width = (upper_bb - lower_bb) / ma_20  # Normalized width
        
        # BB Squeeze: width < 20th percentile of last 50 periods
        bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=20).quantile(0.20).values
        squeeze_condition = bb_width < bb_width_percentile
        
        # Breakout conditions
        breakout_up = close > upper_bb
        breakout_down = close < lower_bb
    else:
        ma_20 = np.full(n, np.nan)
        upper_bb = np.full(n, np.nan)
        lower_bb = np.full(n, np.nan)
        bb_width = np.full(n, np.nan)
        squeeze_condition = np.zeros(n, dtype=bool)
        breakout_up = np.zeros(n, dtype=bool)
        breakout_down = np.zeros(n, dtype=bool)
    
    # Volume confirmation: volume > 1.5x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ma_20[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: BB squeeze breakout above upper BB AND price > weekly R3 AND volume spike
            if squeeze_condition[i-1] and breakout_up[i] and close[i] > camarilla_r3_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: BB squeeze breakout below lower BB AND price < weekly S3 AND volume spike
            elif squeeze_condition[i-1] and breakout_down[i] and close[i] < camarilla_s3_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below weekly S3 (mean reversion) OR squeeze breaks down
            if close[i] < camarilla_s3_aligned[i] or not squeeze_condition[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above weekly R3 (mean reversion) OR squeeze breaks down
            if close[i] > camarilla_r3_aligned[i] or not squeeze_condition[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals