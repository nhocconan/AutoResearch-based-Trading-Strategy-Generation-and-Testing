#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h HMA(21) trend filter and volume confirmation (1.5x avg)
# Uses 1h primary timeframe for Camarilla breakout signals
# 4h HMA(21) confirms medium-term trend direction (avoids counter-trend trades)
# Volume confirmation (1.5x 20-period average) ensures strong participation
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods
# Discrete position sizing (0.20) minimizes fee churn while maintaining profitability
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# Camarilla levels provide precise intraday support/resistance, 4h HMA adds robust trend filter
# Works in both bull and bear markets by only trading in direction of 4h trend

name = "1h_Camarilla_R3S3_Breakout_4hHMA21_Trend_Volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for HMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    # Calculate 4h HMA(21)
    close_4h = pd.Series(df_4h['close'])
    half_length = 21 // 2
    sqrt_length = int(np.sqrt(21))
    
    wma_half = close_4h.rolling(window=half_length, min_periods=half_length).mean()
    wma_full = close_4h.rolling(window=21, min_periods=21).mean()
    raw_hma = 2 * wma_half - wma_full
    hma_4h = raw_hma.rolling(window=sqrt_length, min_periods=sqrt_length).mean().values
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h Camarilla levels (based on previous day's OHLC)
    # Camarilla R3 = close + (high - low) * 1.1/4
    # Camarilla S3 = close - (high - low) * 1.1/4
    # We need daily OHLC, so get 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    camarilla_r3 = daily_close + (daily_high - daily_low) * 1.1 / 4
    camarilla_s3 = daily_close - (daily_high - daily_low) * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe (they change only at daily boundaries)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for calculations)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Check for NaN values in indicators
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(hma_4h_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Camarilla breakout long: close > R3
            # Camarilla breakout short: close < S3
            breakout_long = close[i] > camarilla_r3_aligned[i]
            breakout_short = close[i] < camarilla_s3_aligned[i]
            
            # 4h HMA trend filter: close > HMA for longs, close < HMA for shorts
            hma_long = close[i] > hma_4h_aligned[i]
            hma_short = close[i] < hma_4h_aligned[i]
            
            if breakout_long and hma_long and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            elif breakout_short and hma_short and volume_spike[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Camarilla reversal (close < S3) or trend reversal
            if close[i] < camarilla_s3_aligned[i] or close[i] < hma_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: Camarilla breakout (close > R3) or trend reversal
            if close[i] > camarilla_r3_aligned[i] or close[i] > hma_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals