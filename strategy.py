#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA50 trend filter and volume confirmation.
# Uses Camarilla pivot levels from daily timeframe for structure, breaks above R1 or below S1 for entry,
# confirmed by 1d EMA50 trend and volume spike (>2.0x 20-bar MA). Designed for 4h timeframe to achieve
# 75-200 total trades over 4 years (19-50/year) with discrete sizing (0.25). Works in both bull and bear
# markets via volatility-based breakouts and tight entry conditions requiring confluence of structure,
# trend, and volume. Focus on BTC/ETH as primary targets.

name = "4h_Camarilla_R1_S1_Breakout_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily HTF data for Camarilla pivots and EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (based on previous day's OHLC)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla equations for R1/S1 (tighter levels than R3/S3)
    R1 = prev_close + 1.1 * (prev_high - prev_low) / 12
    S1 = prev_close - 1.1 * (prev_high - prev_low) / 12
    R2 = prev_close + 1.1 * (prev_high - prev_low) / 6
    S2 = prev_close - 1.1 * (prev_high - prev_low) / 6
    
    # Align daily levels to 4h timeframe (wait for completed daily bar)
    R1_4h = align_htf_to_ltf(prices, df_1d, R1)
    S1_4h = align_htf_to_ltf(prices, df_1d, S1)
    R2_4h = align_htf_to_ltf(prices, df_1d, R2)
    S2_4h = align_htf_to_ltf(prices, df_1d, S2)
    
    # Daily EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup for all indicators
    start_idx = 50  # Need 50 for EMA + 1 for Camarilla shift
    
    for i in range(start_idx, n):
        if np.isnan(R1_4h[i]) or np.isnan(S1_4h[i]) or np.isnan(ema_50_4h[i]) or np.isnan(volume_ma_20[i]):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above R1, above daily EMA50, and volume confirmation
            if curr_high > R1_4h[i] and curr_close > ema_50_4h[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Price breaks below S1, below daily EMA50, and volume confirmation
            elif curr_low < S1_4h[i] and curr_close < ema_50_4h[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on close below R1 (failed breakout) or below daily EMA50
            if curr_close < R1_4h[i] or curr_close < ema_50_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on close above S1 (failed breakdown) or above daily EMA50
            if curr_close > S1_4h[i] or curr_close > ema_50_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals