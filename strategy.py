#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot Breakout with 1d Trend Filter and Volume Confirmation
# Uses Camarilla pivot levels (S3/S4 for long, R3/R4 for short) from 1d for entry signals
# 1d EMA (50) provides trend direction filter to avoid counter-trend trades
# Volume confirmation (>1.5x average) ensures institutional participation
# Designed to work in both bull and bear markets by trading reversals at key pivot levels
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for pivot levels and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA (50) for trend direction
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Camarilla pivot levels from previous 1d candle
    # R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), S3 = C - ((H-L)*1.1/4), S4 = C - ((H-L)*1.1/2)
    # Using previous day's data to avoid look-ahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Calculate pivot levels using previous day's data
    camarilla_s4 = close_1d_prev - ((high_1d - low_1d) * 1.1 / 2)
    camarilla_s3 = close_1d_prev - ((high_1d - low_1d) * 1.1 / 4)
    camarilla_r3 = close_1d_prev + ((high_1d - low_1d) * 1.1 / 4)
    camarilla_r4 = close_1d_prev + ((high_1d - low_1d) * 1.1 / 2)
    
    # Align pivot levels to 12h timeframe (using previous day's levels for current day trading)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    
    # Volume confirmation: volume > 1.5x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for EMA and pivot calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend filter: only trade in direction of 1d EMA
        above_ema = price > ema_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above S3 with volume filter and above 1d EMA (mean reversion from oversold)
            if price > camarilla_s3_aligned[i] and vol > 1.5 * avg_vol[i] and above_ema:
                position = 1
                signals[i] = position_size
            # Short: price breaks below R3 with volume filter and below 1d EMA (mean reversion from overbought)
            elif price < camarilla_r3_aligned[i] and vol > 1.5 * avg_vol[i] and not above_ema:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches S4 (target) or breaks below S3 (stop) or reverses trend
            if price >= camarilla_s4_aligned[i] or price < camarilla_s3_aligned[i] or price < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches R4 (target) or breaks above R3 (stop) or reverses trend
            if price <= camarilla_r4_aligned[i] or price > camarilla_r3_aligned[i] or price > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Camarilla_Pivot_Breakout_1dEMA_Volume"
timeframe = "12h"
leverage = 1.0