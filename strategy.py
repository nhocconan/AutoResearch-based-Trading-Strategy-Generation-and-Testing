#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot R3/S3 breakout with 1w EMA34 trend filter and ATR volatility filter
# Uses Camarilla levels for structure, 1w EMA34 for trend alignment (reduces whipsaw in bear markets)
# ATR(14) > 20-bar average ATR filters for sufficient volatility to avoid choppy markets
# Discrete sizing 0.25 to limit fee drag; target 30-100 trades over 4 years
# Proven pattern: Camarilla breakouts with volume/volatility confirmation work on BTC/ETH in both bull/bear

name = "1d_Camarilla_R3S3_1wEMA34_ATRFilter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 20 or len(df_1w) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d Camarilla levels (based on previous day's range)
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    prev_close = pd.Series(close_1d).shift(1).values
    prev_high = pd.Series(high_1d).shift(1).values
    prev_low = pd.Series(low_1d).shift(1).values
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Calculate 1w EMA34 trend filter
    close_1w_series = pd.Series(close_1w)
    ema34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate ATR(14) for volatility filter
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d).shift(1) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d).shift(1) - pd.Series(close_1d).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    avg_atr_20 = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    volatility_filter = atr > (1.2 * avg_atr_20)  # Require above-average volatility
    
    # Align HTF indicators to 1d timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    volatility_filter_aligned = align_htf_to_ltf(prices, df_1d, volatility_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(volatility_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > R3 AND uptrend (price > EMA34) AND sufficient volatility
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema34_1w_aligned[i] and volatility_filter_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price < S3 AND downtrend (price < EMA34) AND sufficient volatility
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema34_1w_aligned[i] and volatility_filter_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retests S3 from above (trend reversal)
            if close[i] <= camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retests R3 from below (trend reversal)
            if close[i] >= camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals