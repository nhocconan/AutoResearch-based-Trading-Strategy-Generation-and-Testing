#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R Extreme + Daily Trend Filter + Volume Spike
# Williams %R identifies overbought/oversold conditions that often reverse in ranging markets
# Daily EMA50 filter ensures we trade mean reversions in direction of higher timeframe trend
# Volume spike confirms exhaustion of current move
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend)
# Target: 20-40 trades/year (80-160 total over 4 years)

name = "4h_WilliamsR_DailyTrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for daily calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Williams %R (14) on 4h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where((highest_high - lowest_low) != 0, 
                         -100 * (highest_high - close) / (highest_high - lowest_low), 
                         -50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(60, 14, 20)  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(williams_r[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_wr = williams_r[i]
        curr_volume_spike = volume_spike[i]
        curr_ema50_1d = ema50_1d_aligned[i]
        
        # Determine trend regime from daily EMA50
        bullish_regime = curr_close > curr_ema50_1d
        bearish_regime = curr_close < curr_ema50_1d
        
        if position == 0:  # Flat - look for new entries
            # Look for mean reversion from extremes with volume confirmation
            if curr_volume_spike:
                # Long setup: oversold in bullish regime
                if bullish_regime and curr_wr < -80:
                    signals[i] = 0.25
                    position = 1
                # Short setup: overbought in bearish regime
                elif bearish_regime and curr_wr > -20:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: Williams %R returns to neutral territory OR reverse signal
            if curr_wr > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: Williams %R returns to neutral territory OR reverse signal
            if curr_wr < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals