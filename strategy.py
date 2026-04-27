# 4H_CAMARILLA_REVERSAL_1D_CONFIRM
# Hypothesis: 4-hour reversals at Camarilla R4/S4 levels with 1-day volume confirmation and 1-day trend filter.
# Works in bull/bear markets: uses daily trend as regime filter and volume to confirm reversal strength.
# Target: 20-40 trades/year per symbol (80-160 total over 4 years) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4-hour data for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate 4-hour Camarilla levels (using previous bar's OHLC)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Use previous bar's range for current bar's Camarilla levels
    range_4h = pd.Series(high_4h).shift(1) - pd.Series(low_4h).shift(1)
    close_prev = pd.Series(close_4h).shift(1)
    
    # Camarilla levels: based on previous bar's range
    S1 = close_prev + (range_4h * 1.0 / 12)
    S2 = close_prev + (range_4h * 2.0 / 12)
    S3 = close_prev + (range_4h * 3.0 / 12)
    S4 = close_prev + (range_4h * 4.0 / 12)
    R1 = close_prev + (range_4h * 11.0 / 12)
    R2 = close_prev + (range_4h * 12.0 / 12)
    R3 = close_prev + (range_4h * 13.0 / 12)
    R4 = close_prev + (range_4h * 14.0 / 12)
    
    # Align Camarilla levels to 4-hour timeframe
    S1_aligned = align_htf_to_ltf(prices, df_4h, S1.values)
    S2_aligned = align_htf_to_ltf(prices, df_4h, S2.values)
    S3_aligned = align_htf_to_ltf(prices, df_4h, S3.values)
    S4_aligned = align_htf_to_ltf(prices, df_4h, S4.values)
    R1_aligned = align_htf_to_ltf(prices, df_4h, R1.values)
    R2_aligned = align_htf_to_ltf(prices, df_4h, R2.values)
    R3_aligned = align_htf_to_ltf(prices, df_4h, R3.values)
    R4_aligned = align_htf_to_ltf(prices, df_4h, R4.values)
    
    # Get daily data for volume filter and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day volume MA(20)
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 1-day EMA(25) for trend
    close_1d = df_1d['close'].values
    ema_25_1d = pd.Series(close_1d).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema_25_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_25_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need Camarilla levels, volume MA, and daily EMA
    start_idx = max(1, 20, 25)  # max of lookbacks
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(S4_aligned[i]) or np.isnan(R4_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(ema_25_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Current 4-hour price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        trend_1d = ema_25_1d_aligned[i]
        
        # Current Camarilla levels
        S4_now = S4_aligned[i]
        R4_now = R4_aligned[i]
        
        # Volume filter: volume > 1.3x 1-day average
        vol_filter = vol_now > 1.3 * vol_ma
        
        # Entry conditions: Camarilla reversal with volume and daily trend alignment
        if position == 0:
            # Long: price at S4 with volume + daily uptrend
            if price_now <= S4_now and vol_filter and price_now > trend_1d:
                signals[i] = size
                position = 1
            # Short: price at R4 with volume + daily downtrend
            elif price_now >= R4_now and vol_filter and price_now < trend_1d:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches S3 or daily trend turns down
            S3_now = S3_aligned[i]
            if price_now >= S3_now or price_now < trend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price reaches R3 or daily trend turns up
            R3_now = R3_aligned[i]
            if price_now <= R3_now or price_now > trend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_CamarillaS4R4_Reversal_1dVolume_1dTrend"
timeframe = "4h"
leverage = 1.0