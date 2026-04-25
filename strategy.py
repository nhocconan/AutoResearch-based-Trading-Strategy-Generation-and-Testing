#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike
Hypothesis: Camarilla R3/S3 breakout on 4h with 1d EMA34 trend filter and volume confirmation.
Long when price breaks above R3 in uptrend (close > daily EMA34) with volume spike.
Short when price breaks below S3 in downtrend (close < daily EMA34) with volume spike.
Exit when price re-enters Camarilla H3/L3 levels or trend reverses.
Designed for low trade frequency (target: 75-200 total trades over 4 years) and robustness in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla calculations (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla pivot levels (based on previous day's OHLC)
    # Typical timeframe: daily, but we apply to 4h bars using previous daily pivot
    # We'll calculate daily pivots first, then align to 4h
    
    # Get 1d data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels
    # Formula: 
    # H4 = close + 1.5*(high - low)
    # H3 = close + 1.25*(high - low)
    # H2 = close + 1.166*(high - low)
    # H1 = close + 1.0833*(high - low)
    # L1 = close - 1.0833*(high - low)
    # L2 = close - 1.166*(high - low)
    # L3 = close - 1.25*(high - low)
    # L4 = close - 1.5*(high - low)
    # Where (high, low, close) are from previous day
    
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Avoid look-ahead: use shifted values (previous day's data)
    high_low_diff = prev_high - prev_low
    
    H3 = prev_close + 1.25 * high_low_diff
    L3 = prev_close - 1.25 * high_low_diff
    H1 = prev_close + 1.0833 * high_low_diff
    L1 = prev_close - 1.0833 * high_low_diff
    
    # Align daily Camarilla levels to 4h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H1_aligned = align_htf_to_ltf(prices, df_1d, H1)
    L1_aligned = align_htf_to_ltf(prices, df_1d, L1)
    
    # Get weekly data for trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA34 for trend direction
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend = ema_34_1w_aligned[i]
        
        if position == 0:
            # Regime-based entry logic
            if close[i] > ema_trend:  # Uptrend regime (weekly)
                # Long: break above H3 with volume spike
                long_signal = (close[i] > H3_aligned[i]) and vol_spike[i]
                # Short: break below L3 only if extreme volume spike (counter-trend fade)
                short_signal = (close[i] < L3_aligned[i]) and vol_spike[i] and (volume[i] > (4.0 * vol_ma_20[i]))
            else:  # Downtrend regime (weekly)
                # Short: break below L3 with volume spike
                short_signal = (close[i] < L3_aligned[i]) and vol_spike[i]
                # Long: break above H3 only if extreme volume spike (counter-trend fade)
                long_signal = (close[i] > H3_aligned[i]) and vol_spike[i] and (volume[i] > (4.0 * vol_ma_20[i]))
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions: re-enter H1/L1 zone or trend reversal
            exit_signal = (close[i] < H1_aligned[i]) or (close[i] < ema_trend * 0.99)
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions: re-enter H1/L1 zone or trend reversal
            exit_signal = (close[i] > L1_aligned[i]) or (close[i] > ema_trend * 1.01)
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0