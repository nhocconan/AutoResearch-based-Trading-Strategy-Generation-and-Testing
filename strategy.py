#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h EMA trend filter and volume confirmation
# Camarilla pivots (R3/S3) identify institutional support/resistance levels on 4h.
# Breakouts above R3 or below S3 with volume spike and 4h EMA trend alignment capture strong moves.
# Uses 1h for precise entry timing, 4h for trend direction and pivot levels.
# Session filter (08-20 UTC) reduces noise. Target: 15-30 trades/year to minimize fee drag.
# Works in bull (breakouts with volume) and bear (mean reversion via opposite breakouts after volatility expansion).

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (precompute once)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h HTF data for Camarilla pivots and EMA trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 4h Camarilla pivots (R3, S3) - based on previous 4h bar's range
    # Typical Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    # Using previous completed 4h bar to avoid look-ahead
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_shift = df_4h['close'].shift(1).values  # Previous 4h bar close
    rang
    camarilla_range = (high_4h - low_4h) * 1.1 / 2
    r3_4h = close_4h_shift + camarilla_range
    s3_4h = close_4h_shift - camarilla_range
    
    # AlCamarilla levels to 1h
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    
    # Volume confirmation: current volume > 1.8 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 50  # Need sufficient history for EMA and volume MA
    
    for i in range(start_idx, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(r3_4h_aligned[i]) or 
            np.isnan(s3_4h_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Trend filter: price above/below 4h EMA50
        uptrend = curr_close > ema_50_4h_aligned[i]
        downtrend = curr_close < ema_50_4h_aligned[i]
        
        # Camarilla breakout conditions
        breakout_up = curr_close > r3_4h_aligned[i]   # Break above R3
        breakout_down = curr_close < s3_4h_aligned[i]  # Break below S3
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: breakout above R3, volume spike, uptrend
            if breakout_up and vol_spike and uptrend:
                signals[i] = 0.20
                position = 1
            # Short: breakout below S3, volume spike, downtrend
            elif breakout_down and vol_spike and downtrend:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on break below S3 or trend reversal
            if curr_close < s3_4h_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit on break above R3 or trend reversal
            if curr_close > r3_4h_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals