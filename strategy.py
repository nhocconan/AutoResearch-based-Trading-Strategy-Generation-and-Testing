#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout + 1d volume confirmation
# - Camarilla levels calculated from 1d OHLC: R3, R4, S3, S4 act as institutional support/resistance
# - Long when price breaks above R4 with volume > 1.5x 20-period 1d volume average (breakout continuation)
# - Short when price breaks below S4 with volume > 1.5x 20-period 1d volume average (breakdown continuation)
# - Fade trades at R3/S3 with volume < 0.8x 20-period average (mean reversion in range)
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within 6h fee drag limits
# - Camarilla pivots work in both bull (breakouts at R4/S4) and bear (mean reversion at R3/S3) markets
# - 1d HTF provides reliable structure, 6h timeframe balances frequency and cost

name = "6h_1d_camarilla_breakout_v1"
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
    
    # Load 1d data ONCE before loop for Camarilla pivots and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Pre-compute 1d Camarilla levels (based on previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar (using previous day's data)
    camarilla_r4 = np.full(len(df_1d), np.nan)
    camarilla_r3 = np.full(len(df_1d), np.nan)
    camarilla_s3 = np.full(len(df_1d), np.nan)
    camarilla_s4 = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        # Use previous day's OHLC to avoid look-ahead
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_ = prev_high - prev_low
        
        camarilla_r4[i] = prev_close + range_ * 1.1 / 2
        camarilla_r3[i] = prev_close + range_ * 1.1 / 4
        camarilla_s3[i] = prev_close - range_ * 1.1 / 4
        camarilla_s4[i] = prev_close - range_ * 1.1 / 2
    
    # Pre-compute 1d volume SMA (20-period)
    volume_1d = df_1d['volume'].values
    volume_series = pd.Series(volume_1d)
    volume_sma_20_1d = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(volume_sma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Camarilla levels (from previous 1d bar)
        r4 = camarilla_r4_aligned[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        s4 = camarilla_s4_aligned[i]
        vol_ma = volume_sma_20_aligned[i]
        
        # Volume conditions
        vol_expansion = volume_current > 1.5 * vol_ma  # Strong volume for breakouts
        vol_contraction = volume_current < 0.8 * vol_ma  # Weak volume for mean reversion
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price breaks above R4 with expanding volume
        if price_close > r4 and vol_expansion:
            enter_long = True
        
        # Short breakdown: price breaks below S4 with expanding volume
        if price_close < s4 and vol_expansion:
            enter_short = True
        
        # Long mean reversion: price rejects at S3 with contracting volume
        if price_close < s3 and price_low >= s4 and vol_contraction and position != 1:
            enter_long = True
        
        # Short mean reversion: price rejects at R3 with contracting volume
        if price_close > r3 and price_high <= r4 and vol_contraction and position != -1:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long: price breaks below R3 (failed breakout) or reaches R4 (profit target)
            exit_long = (price_close < r3) or (price_close >= r4 and vol_expansion)
        elif position == -1:
            # Exit short: price breaks above S3 (failed breakdown) or reaches S4 (profit target)
            exit_short = (price_close > s3) or (price_close <= s4 and vol_expansion)
        
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