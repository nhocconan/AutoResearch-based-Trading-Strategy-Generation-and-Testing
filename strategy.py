#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) Breakout with Weekly Trend Filter and Volume Confirmation
# Donchian channel breakouts capture strong momentum moves in both bull and bear markets
# Weekly EMA50 trend filter ensures we trade breakouts in direction of higher timeframe trend
# Volume confirmation validates breakout strength and reduces false signals
# Target: 20-50 total trades over 4 years (5-12/year) to minimize fee drag

name = "1d_Donchian20_WeeklyTrend_VolumeConfirm_v1"
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
    
    # Calculate Donchian Channel (20) on daily data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(60, 20, 20)  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_donchian_upper = donchian_upper[i]
        curr_donchian_lower = donchian_lower[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema50_1w = ema50_1w_aligned[i]
        
        # Determine trend regime from weekly EMA50
        bullish_regime = curr_close > curr_ema50_1w
        bearish_regime = curr_close < curr_ema50_1w
        
        if position == 0:  # Flat - look for new entries
            # Look for Donchian breakout with volume confirmation
            if curr_volume_confirm:
                # Bullish breakout: price breaks above upper Donchian in bullish regime
                if bullish_regime and curr_close > curr_donchian_upper:
                    signals[i] = 0.30
                    position = 1
                # Bearish breakout: price breaks below lower Donchian in bearish regime
                elif bearish_regime and curr_close < curr_donchian_lower:
                    signals[i] = -0.30
                    position = -1
        
        elif position == 1:  # Long position - exit when price touches opposite band
            if curr_close < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position - exit when price touches opposite band
            if curr_close > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals