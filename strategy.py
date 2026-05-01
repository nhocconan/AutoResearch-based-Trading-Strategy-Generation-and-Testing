#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h EMA50 trend filter and volume spike (>1.5x 24-bar MA)
# Uses 4h EMA50 for higher-timeframe trend alignment to reduce whipsaws.
# Volume spike confirms institutional participation. Discrete sizing (0.20) minimizes fee churn.
# Session filter (08-20 UTC) reduces noise trades.
# Target: 60-150 total trades over 4 years (15-37/year) to stay within fee drag limits.
# Camarilla pivot levels calculated from prior day's range provide institutional reference points.

name = "1h_Camarilla_R1_S1_Breakout_4hEMA50_Trend_VolumeSpike_v1"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h HTF data for EMA calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA(50) on 4h close
    ema_4h_50 = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA to 1h timeframe
    ema_4h_50_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_50)
    
    # Prior day's high, low, close for Camarilla pivot calculation
    # Use 1d data from mtf_data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate prior day's OHLC for Camarilla levels
    # We need the completed prior day's data, so we shift by 1
    prior_high = df_1d['high'].shift(1).values
    prior_low = df_1d['low'].shift(1).values
    prior_close = df_1d['close'].shift(1).values
    
    # Camarilla pivot levels (R1, S1)
    # R1 = Close + (High - Low) * 1.1/12
    # S1 = Close - (High - Low) * 1.1/12
    camarilla_range = prior_high - prior_low
    r1 = prior_close + camarilla_range * 1.1 / 12
    s1 = prior_close - camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe (need extra delay for prior day's data)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1, additional_delay_bars=1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1, additional_delay_bars=1)
    
    # Volume confirmation: current volume > 1.5 * 24-period average volume
    volume_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (volume_ma_24 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 30  # Need 24 for volume MA, plus buffer
    
    for i in range(start_idx, n):
        if (np.isnan(ema_4h_50_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_ma_24[i])):
            signals[i] = 0.0
            continue
        
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R1 with volume spike and above 4h EMA50
            if curr_close > r1_aligned[i] and curr_high > r1_aligned[i] and vol_spike and curr_close > ema_4h_50_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 with volume spike and below 4h EMA50
            elif curr_close < s1_aligned[i] and curr_low < s1_aligned[i] and vol_spike and curr_close < ema_4h_50_aligned[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on price below S1 or below 4h EMA50
            if curr_close < s1_aligned[i] or curr_close < ema_4h_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit on price above R1 or above 4h EMA50
            if curr_close > r1_aligned[i] or curr_close > ema_4h_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals