#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot reversals with 1d trend filter
# - Camarilla levels from 1d: R3/S3 for mean reversion, R4/S4 for breakout confirmation
# - Long when price rejects S3 (close > S3 after touching <= S3) with 1d uptrend (close > EMA50)
# - Short when price rejects R3 (close < R3 after touching >= R3) with 1d downtrend (close < EMA50)
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within 6h fee limits
# - Works in bull (R3 rejections in uptrend) and bear (S3 rejections in downtrend) markets
# - 1d EMA50 filter ensures we only take reversals in the direction of higher timeframe trend

name = "6h_1d_camarilla_reversal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Calculate 1d Camarilla levels (based on previous day's range)
    # Using previous day's high, low, close to avoid look-ahead
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate pivot and ranges
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    # Camarilla levels
    r3 = pivot + (range_val * 1.1 / 2.0)  # Level where reversal often occurs
    s3 = pivot - (range_val * 1.1 / 2.0)
    r4 = pivot + (range_val * 1.1)        # Breakout level
    s4 = pivot - (range_val * 1.1)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    for i in range(50, n):  # Start after 50-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        
        # Rejection conditions: price touches level then reverses
        touched_s3 = price_low <= s3_aligned[i]
        rejected_s3 = price_close > s3_aligned[i] and touched_s3
        
        touched_r3 = price_high >= r3_aligned[i]
        rejected_r3 = price_close < r3_aligned[i] and touched_r3
        
        # Trend filter from 1d EMA50
        uptrend = price_close > ema50_aligned[i]
        downtrend = price_close < ema50_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: S3 rejection in uptrend (mean reversion)
        if rejected_s3 and uptrend:
            enter_long = True
        
        # Short: R3 rejection in downtrend (mean reversion)
        if rejected_r3 and downtrend:
            enter_short = True
        
        # Exit conditions: opposite rejection or breakout beyond R4/S4
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long on R3 rejection in downtrend or break above R4
            exit_long = (rejected_r3 and downtrend) or (price_close > r4_aligned[i])
        elif position == -1:
            # Exit short on S3 rejection in uptrend or break below S4
            exit_short = (rejected_s3 and uptrend) or (price_close < s4_aligned[i])
        
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