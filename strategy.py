#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot Reversal with 1d Trend Filter and Volume Confirmation
# Camarilla pivot levels (R3/S3) act as strong support/resistance in ranging markets.
# Price rejection at these levels with volume confirmation indicates reversal.
# 1d EMA50 filter ensures alignment with higher timeframe trend to avoid counter-trend trades.
# ATR-based stop loss manages risk. Designed for 4h with selective entries (<50/year).
# Works in both bull (buy S3 dips in uptrend) and bear (sell R3 rallies in downtrend).

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 50-period EMA on 1d timeframe for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Camarilla pivot levels from previous day
    # Typical price = (high + low + close) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    typical_price_vals = typical_price.values
    
    # Camarilla levels: 
    # R4 = close + ((high - low) * 1.500)
    # R3 = close + ((high - low) * 1.250)
    # R2 = close + ((high - low) * 1.166)
    # R1 = close + ((high - low) * 1.083)
    # S1 = close - ((high - low) * 1.083)
    # S2 = close - ((high - low) * 1.166))
    # S3 = close - ((high - low) * 1.250)
    # S4 = close - ((high - low) * 1.500)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    camarilla_calc = ((high_1d - low_1d) * 1.250)  # For S3/R3
    s3_level = close_1d_vals - camarilla_calc
    r3_level = close_1d_vals + camarilla_calc
    
    # Align S3 and R3 to 4h timeframe (using previous day's levels)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_level)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_level)
    
    # Calculate ATR for stop loss
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: volume > 1.3x 20-period average
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(60, n):
        # Skip if NaN in indicators
        if np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]) or \
           np.isnan(ema50_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend
        is_uptrend = close[i] > ema50_1d_aligned[i]
        is_downtrend = close[i] < ema50_1d_aligned[i]
        
        # Volume confirmation
        has_volume = vol_filter[i]
        
        price = close[i]
        
        if position == 0:
            # Long entry: price rejects S3 (bounces off) + uptrend + volume
            # Rejection: price > S3 AND price was <= S3 in previous bar
            long_rejection = (price > s3_aligned[i]) and (prices['close'].iloc[i-1] <= s3_aligned[i-1]) if i > 0 else False
            
            # Short entry: price rejects R3 (fails to break) + downtrend + volume
            # Rejection: price < R3 AND price was >= R3 in previous bar
            short_rejection = (price < r3_aligned[i]) and (prices['close'].iloc[i-1] >= r3_aligned[i-1]) if i > 0 else False
            
            if long_rejection and is_uptrend and has_volume:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_rejection and is_downtrend and has_volume:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: stop loss or R3 test (take profit near resistance)
            stop_loss = entry_price - 2.0 * atr[i]
            take_profit = price >= r3_aligned[i]  # Take profit near R3 level
            
            if stop_loss <= 0 or price <= stop_loss or take_profit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: stop loss or S3 test (take profit near support)
            stop_loss = entry_price + 2.0 * atr[i]
            take_profit = price <= s3_aligned[i]  # Take profit near S3 level
            
            if stop_loss <= 0 or price >= stop_loss or take_profit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_S3R3_1dTrendFilter_Volume"
timeframe = "4h"
leverage = 1.0