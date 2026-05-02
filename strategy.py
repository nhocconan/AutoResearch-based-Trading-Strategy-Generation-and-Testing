#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation
# Uses 6h timeframe for signal generation with Camarilla pivot breakouts from R3/S3 levels
# 12h EMA(50) determines primary trend direction - avoids counter-trend trades
# Volume spike (1.8x 24-period average) ensures strong participation
# Discrete position sizing (0.25) minimizes fee drag while maintaining profitability
# Target: 75-150 total trades over 4 years = 19-38/year for 6h timeframe
# Camarilla pivots provide mathematically derived support/resistance levels
# Works in both bull and bear markets by only taking trades aligned with 12h trend

name = "6h_Camarilla_R3S3_Breakout_12hEMA50_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h EMA(50) for trend determination
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla pivot levels from prior 1d bar (using daily data for pivot calculation)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Prior 1d bar's OHLC for Camarilla calculation
    prior_close = np.roll(close_1d, 1)
    prior_high = np.roll(high_1d, 1)
    prior_low = np.roll(low_1d, 1)
    prior_close[0] = np.nan
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    
    # Camarilla calculations: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    # R4 = C + (H-L)*1.1, S4 = C - (H-L)*1.1
    rang = prior_high - prior_low
    camarilla_r3 = prior_close + rang * 1.1 / 2
    camarilla_s3 = prior_close - rang * 1.1 / 2
    camarilla_r4 = prior_close + rang * 1.1
    camarilla_s4 = prior_close - rang * 1.1
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume confirmation (1.8x 24-period average)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Close > R3 + volume spike + close > 12h EMA50 (bullish trend)
            if close[i] > r3_aligned[i] and volume_spike[i] and close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close < S3 + volume spike + close < 12h EMA50 (bearish trend)
            elif close[i] < s3_aligned[i] and volume_spike[i] and close[i] < ema_50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close < S3 or close < 12h EMA50 (trend reversal)
            if close[i] < s3_aligned[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close > R3 or close > 12h EMA50 (trend reversal)
            if close[i] > r3_aligned[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals