#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R with Weekly Trend Filter and Volume Confirmation
# Williams %R identifies overbought/oversold conditions for mean reversion entries
# Weekly trend filter (EMA50) ensures we trade reversals in direction of higher timeframe trend
# Volume spike confirms reversal strength
# Works in all regimes: mean reversion effective in ranging markets, trend filter avoids counter-trend trades
# Target: 20-40 trades/year (80-160 total over 4 years)

name = "1d_WilliamsR_WeeklyTrend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for weekly calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Williams %R (14) on 1d data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where((highest_high - lowest_low) != 0, 
                          -100 * ((highest_high - close) / (highest_high - lowest_low)), 
                          -50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(60, 14, 20)  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(williams_r[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema50_1w = ema50_1w_aligned[i]
        
        # Determine trend regime from weekly EMA50
        bullish_regime = curr_close > curr_ema50_1w
        bearish_regime = curr_close < curr_ema50_1w
        
        if position == 0:  # Flat - look for new entries
            # Look for mean reversion from extremes with volume confirmation
            if curr_volume_confirm:
                # Long entry: oversold (< -80) in bullish weekly regime
                if bullish_regime and curr_williams_r < -80:
                    signals[i] = 0.25
                    position = 1
                # Short entry: overbought (> -20) in bearish weekly regime
                elif bearish_regime and curr_williams_r > -20:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: Williams %R returns to neutral range (> -50) OR weekly trend changes
            if curr_williams_r > -50 or curr_close < curr_ema50_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: Williams %R returns to neutral range (< -50) OR weekly trend changes
            if curr_williams_r < -50 or curr_close > curr_ema50_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals