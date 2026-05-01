#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Camarilla pivots provide precise intraday support/resistance levels. Breakouts from R3/S3 levels with volume confirmation capture strong momentum moves.
# 4h EMA50 ensures alignment with medium-term trend to avoid counter-trend trades.
# Session filter (08-20 UTC) reduces noise during low-liquidity periods.
# Target: 60-150 total trades over 4 years (15-37/year). Works in bull (breakouts with volume) and bear (volatility expansion after consolidation).
# Discrete sizing (0.20) minimizes fee churn while maintaining adequate exposure.

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
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h HTF data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA(50) calculation
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla pivots for 1h timeframe using prior day's OHLC
    # Camarilla levels: H4 = C + 1.1*(H-L)/2, L4 = C - 1.1*(H-L)/2
    # R3 = C + 1.1*(H-L)/2, S3 = C - 1.1*(H-L)/2 (same as H4/L4)
    # We need prior day's OHLC - resample to 1d first
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior day's OHLC (shifted by 1 to avoid look-ahead)
    prior_close = df_1d['close'].shift(1).values
    prior_high = df_1d['high'].shift(1).values
    prior_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla R3/S3 levels for prior day
    camarilla_r3 = prior_close + 1.1 * (prior_high - prior_low) / 2
    camarilla_s3 = prior_close - 1.1 * (prior_high - prior_low) / 2
    
    # Align Camarilla levels to 1h timeframe (each level applies to entire following day)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: current volume > 2.0 * 24-period average volume
    volume_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (volume_ma_24 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 100  # Need sufficient history for all calculations
    
    for i in range(start_idx, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(volume_ma_24[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Trend filter: price above/below 4h EMA50
        trend_up = curr_close > ema_50_4h_aligned[i]
        trend_down = curr_close < ema_50_4h_aligned[i]
        
        # Camarilla breakout conditions
        breakout_up = curr_close > camarilla_r3_aligned[i]  # Break above R3
        breakout_down = curr_close < camarilla_s3_aligned[i]  # Break below S3
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Camarilla breakout up, volume spike, uptrend
            if breakout_up and vol_spike and trend_up:
                signals[i] = 0.20
                position = 1
            # Short: Camarilla breakout down, volume spike, downtrend
            elif breakout_down and vol_spike and trend_down:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Camarilla breakdown or trend reversal
            if curr_close < camarilla_s3_aligned[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit on Camarilla breakout or trend reversal
            if curr_close > camarilla_r3_aligned[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals