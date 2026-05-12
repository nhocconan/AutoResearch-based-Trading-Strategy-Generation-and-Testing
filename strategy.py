#!/usr/bin/env python3
# 1H_MULTI_TIMEFRAME_CAMARILLA_BREAKOUT_VOLUME_FILTER
# Hypothesis: Camarilla pivot levels on 4h and daily timeframes provide institutional support/resistance levels.
# Combined with volume spike confirmation on 1h and trend filter from daily EMA50.
# Works in both bull and bear markets: breaks above resistance in uptrend, breakdown below support in downtrend.
# Target: 15-35 trades/year on 1h timeframe by using 4h/1d for direction and 1h for precise entry timing.

name = "1H_MULTI_TIMEFRAME_CAMARILLA_BREAKOUT_VOLUME_FILTER"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load higher timeframe data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 10 or len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate Camarilla levels for 4h timeframe (based on previous day's range)
    # Camarilla equations for 4h: use previous day's high, low, close
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Daily high/low/close for Camarilla calculation
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla levels for 4h timeframe using daily OHLC
    # Resistance levels: R1 = C + (H-L)*1.1/12, R2 = C + (H-L)*1.1/6, R3 = C + (H-L)*1.1/4, R4 = C + (H-L)*1.1/2
    # Support levels: S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    camarilla_r4 = daily_close + (daily_high - daily_low) * 1.1 / 2
    camarilla_r3 = daily_close + (daily_high - daily_low) * 1.1 / 4
    camarilla_r2 = daily_close + (daily_high - daily_low) * 1.1 / 6
    camarilla_r1 = daily_close + (daily_high - daily_low) * 1.1 / 12
    camarilla_s1 = daily_close - (daily_high - daily_low) * 1.1 / 12
    camarilla_s2 = daily_close - (daily_high - daily_low) * 1.1 / 6
    camarilla_s3 = daily_close - (daily_high - daily_low) * 1.1 / 4
    camarilla_s4 = daily_close - (daily_high - daily_low) * 1.1 / 2
    
    # Align Camarilla levels to 1h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate EMA50 on daily timeframe for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate volume spike on 1h timeframe (current volume > 2x 20-period average)
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_spike = prices['volume'].values > (2.0 * vol_ma)
    
    # Session filter: 08-20 UTC (already datetime64[ms] in index)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Entry conditions: Camarilla breakout with volume spike and trend filter
        close_price = prices['close'].values[i]
        
        # LONG: Price breaks above R1 with volume spike in uptrend (price > EMA50)
        if (close_price > r1_aligned[i] and 
            volume_spike[i] and 
            close_price > ema50_aligned[i]):
            signals[i] = 0.20
        
        # SHORT: Price breaks below S1 with volume spike in downtrend (price < EMA50)
        elif (close_price < s1_aligned[i] and 
              volume_spike[i] and 
              close_price < ema50_aligned[i]):
            signals[i] = -0.20
        
        else:
            signals[i] = 0.0
    
    return signals