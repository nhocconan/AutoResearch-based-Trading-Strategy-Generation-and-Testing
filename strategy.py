#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Camarilla pivot breakout with 1-week trend filter (EMA10) and volume confirmation
# Long when price breaks above Camarilla R4 (or H4) level AND price > weekly EMA10 AND volume > 1.5x 20-period average
# Short when price breaks below Camarilla S4 (or L4) level AND price < weekly EMA10 AND volume > 1.5x 20-period average
# Exit when price crosses back inside the Camarilla H4-L4 range
# This captures strong trending moves with volume confirmation while avoiding counter-trend trades
# Target: 30-100 total trades over 4 years (7-25/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for EMA10 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate daily Camarilla pivot levels (using previous day's OHLC)
    # We need previous day's data, so we shift by 1
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    
    # Camarilla levels
    R4 = close + (range_val * 1.1 / 2)  # Actually R4 = C + ((H-L) * 1.1/2)
    R3 = close + (range_val * 1.1/4)
    R2 = close + (range_val * 1.1/6)
    R1 = close + (range_val * 1.1/12)
    S1 = close - (range_val * 1.1/12)
    S2 = close - (range_val * 1.1/6)
    S3 = close - (range_val * 1.1/4)
    S4 = close - (range_val * 1.1/2)
    
    # For breakout, we use R4 and S4 as breakout levels
    # But we also consider H4/L4 (which are same as R4/S4 in Camarilla)
    H4 = R4
    L4 = S4
    
    # Calculate weekly EMA10 for trend filter
    close_1w = df_1w['close'].values
    ema10_1w = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema10_1w)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (need at least 1 for previous day, 20 for vol avg)
    start = 21
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(H4[i]) or np.isnan(L4[i]) or 
            np.isnan(ema10_1w_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: breakout above H4 (R4) + above weekly EMA10 + volume confirmation
            if (price > H4[i] and price > ema10_1w_aligned[i] and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: breakdown below L4 (S4) + below weekly EMA10 + volume confirmation
            elif (price < L4[i] and price < ema10_1w_aligned[i] and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls back below L4 (S4) - opposite level
            if price < L4[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price rises back above H4 (R4) - opposite level
            if price > H4[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_Camarilla_1wEMA10_Volume"
timeframe = "1d"
leverage = 1.0