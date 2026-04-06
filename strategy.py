#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot reversal with 1d trend filter and volume confirmation
# Long when price touches Camarilla L3 support AND price > 1d EMA(50) AND volume > 1.5x 20-period average
# Short when price touches Camarilla H3 resistance AND price < 1d EMA(50) AND volume > 1.5x 20-period average
# Exit when price crosses Camarilla pivot (H4/L4 levels)
# Camarilla levels calculated from previous 1d candle
# Target: 50-150 total trades over 4 years (12-37/year) for optimal 12h performance

name = "12h_camarilla_pivot_1d_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Previous day OHLC for Camarilla calculation
    prev_high = np.roll(high, 1)  # 12h to 1d conversion: 2 bars per day
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla levels (based on previous day)
    range_val = prev_high - prev_low
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3
    camarilla_h3 = camarilla_pivot + 1.1 * range_val / 6
    camarilla_l3 = camarilla_pivot - 1.1 * range_val / 6
    camarilla_h4 = camarilla_pivot + 1.1 * range_val / 2
    camarilla_l4 = camarilla_pivot - 1.1 * range_val / 2
    
    # 1-day EMA(50) trend filter
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    daily_close_series = pd.Series(daily_close)
    daily_ema = daily_close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    daily_ema_aligned = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or
            np.isnan(daily_ema_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: price crosses Camarilla H4/L4 levels
        if position == 1:  # long position
            if close[i] < camarilla_l4[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] > camarilla_h4[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend filter and volume confirmation
            # Long: price touches Camarilla L3 support AND price > daily EMA AND volume confirmation
            if (low[i] <= camarilla_l3[i] and low[i-1] > camarilla_l3[i-1] and 
                close[i] > daily_ema_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price touches Camarilla H3 resistance AND price < daily EMA AND volume confirmation
            elif (high[i] >= camarilla_h3[i] and high[i-1] < camarilla_h3[i-1] and 
                  close[i] < daily_ema_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals