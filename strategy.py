#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Camarilla pivot level touch with 1-day trend filter and volume confirmation
# Long when price touches Camarilla L3 level AND price > 1-day EMA200 AND volume > 1.5x 20-period average
# Short when price touches Camarilla H3 level AND price < 1-day EMA200 AND volume > 1.5x 20-period average
# Exit when price touches opposite Camarilla level (L3 for shorts, H3 for longs)
# Uses Camarilla levels for intraday support/resistance, EMA200 for trend filter, volume for confirmation
# Target: 75-200 total trades over 4 years (19-50/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-day data ONCE before loop for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels for each 4h bar using previous day's OHLC
    # We need to map each 4h bar to the prior day's OHLC
    # Create arrays for prior day's high, low, close
    prior_day_high = np.full(n, np.nan)
    prior_day_low = np.full(n, np.nan)
    prior_day_close = np.full(n, np.nan)
    
    # For each 4h bar, find the prior day's OHLC
    for i in range(n):
        current_time = pd.Timestamp(prices['open_time'].iloc[i])
        prior_day = current_time - pd.Timedelta(days=1)
        # Find index of prior day's 4h bar (assuming 6 bars per day)
        prior_idx = i - 6
        if prior_idx >= 0:
            prior_day_high[i] = prices['high'].iloc[prior_idx]
            prior_day_low[i] = prices['low'].iloc[prior_idx]
            prior_day_close[i] = prices['close'].iloc[prior_idx]
    
    # Calculate Camarilla levels: H3, L3
    # H3 = close + (high - low) * 1.1/4
    # L3 = close - (high - low) * 1.1/4
    camarilla_H3 = prior_day_close + (prior_day_high - prior_day_low) * 1.1 / 4
    camarilla_L3 = prior_day_close - (prior_day_high - prior_day_low) * 1.1 / 4
    
    # Calculate 1-day EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 200  # Need enough for EMA200
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(prior_day_high[i]) or np.isnan(prior_day_low[i]) or 
            np.isnan(prior_day_close[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: price touches L3 AND price > 1-day EMA200 AND volume confirmation
            if (price <= camarilla_L3[i] and price > ema200_1d_aligned[i] and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: price touches H3 AND price < 1-day EMA200 AND volume confirmation
            elif (price >= camarilla_H3[i] and price < ema200_1d_aligned[i] and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price touches H3 (opposite level)
            if price >= camarilla_H3[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price touches L3 (opposite level)
            if price <= camarilla_L3[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Camarilla_1dEMA200_Volume"
timeframe = "4h"
leverage = 1.0