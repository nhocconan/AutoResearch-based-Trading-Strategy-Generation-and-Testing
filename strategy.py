#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Uses 4h timeframe for signal generation with Camarilla pivot level breakouts
# Daily trend filter (price > daily EMA34 for longs, < for shorts) ensures alignment with higher timeframe bias
# Volume confirmation (1.8x 20-period average) filters for institutional participation
# Discrete position sizing (0.25) balances profit potential with fee drag minimization
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Camarilla levels provide precise support/resistance, effective in both bull and bear markets

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_Volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h Camarilla levels (based on previous day's OHLC)
    # Camarilla R3 = close + (high - low) * 1.1/4
    # Camarilla S3 = close - (high - low) * 1.1/4
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    # Use previous bar's OHLC to avoid look-ahead (Camarilla based on completed candle)
    prev_high = high_series.shift(1)
    prev_low = low_series.shift(1)
    prev_close = close_series.shift(1)
    
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Daily trend filter: price > daily EMA34 for longs, < for shorts
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation (1.8x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 40
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Break above R3 + volume spike + price > daily EMA34
            if close[i] > camarilla_r3[i] and volume_spike[i] and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below S3 + volume spike + price < daily EMA34
            elif close[i] < camarilla_s3[i] and volume_spike[i] and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close below S3 or price < daily EMA34
            if close[i] < camarilla_s3[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close above R3 or price > daily EMA34
            if close[i] > camarilla_r3[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals