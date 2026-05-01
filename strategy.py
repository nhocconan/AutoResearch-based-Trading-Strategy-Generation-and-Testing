#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme reversal with 1d EMA50 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions. Extreme readings (< -80 or > -20) 
# followed by reversal provide high-probability mean reversion entries in ranging markets.
# 1d EMA50 ensures alignment with daily trend to avoid counter-trend trades during strong moves.
# Volume confirmation filters out false reversals.
# Designed for 6h timeframe to capture multi-day mean reversion cycles with low trade frequency.
# Target: 12-25 trades/year for sustainable performance in both bull and bear markets.

name = "6h_WilliamsR_Extreme_Reversal_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams %R calculation (14-period) on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    
    # Extreme levels: oversold < -80, overbought > -20
    # Reversal conditions: Williams %R crosses back above -80 (from below) for long
    #                      Williams %R crosses back below -20 (from above) for short
    williams_r_prev = np.roll(williams_r, 1)
    williams_r_prev[0] = np.nan
    
    # Bullish reversal: was oversold (< -80) and now crossing above -80
    bullish_reversal = (williams_r_prev <= -80) & (williams_r > -80)
    # Bearish reversal: was overbought (> -20) and now crossing below -20
    bearish_reversal = (williams_r_prev >= -20) & (williams_r < -20)
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(50, lookback)  # Need sufficient history for EMA50 and Williams %R
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(williams_r_prev[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: bullish reversal from oversold, volume spike, not in strong downtrend
            if bullish_reversal[i] and vol_spike and not downtrend:
                signals[i] = 0.25
                position = 1
            # Short: bearish reversal from overbought, volume spike, not in strong uptrend
            elif bearish_reversal[i] and vol_spike and not uptrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on bearish reversal or trend reversal to downtrend
            if bearish_reversal[i] or downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on bullish reversal or trend reversal to uptrend
            if bullish_reversal[i] or uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals