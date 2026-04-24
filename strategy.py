#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H4/L4 breakout with 1d HMA21 trend filter and volume spike confirmation.
- Long when price breaks above H4 AND 1d close > 1d HMA21 (bullish regime)
- Short when price breaks below L4 AND 1d close < 1d HMA21 (bearish regime)
- Volume confirmation: current volume > 1.8 * 24-period average volume (strong spike)
- Exit on opposite Camarilla level (L4 for long exit, H4 for short exit)
- Uses 12h primary with 1d HTF to target 50-150 trades over 4 years (12-37/year)
- HMA21 provides smoother trend than EMA, reducing whipsaw in choppy markets
- Camarilla H4/L4 levels offer stronger breakout signals than H3/L3
- Designed to capture strong momentum moves with regime and volume filters
- Signal size: 0.25 discrete levels to minimize fee churn
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivot levels (based on previous bar's OHLC)
    # H4 = close + 1.1 * (high - low) * 1.125 / 2
    # L4 = close - 1.1 * (high - low) * 1.125 / 2
    # Using previous bar's OHLC to avoid look-ahead
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    camarilla_h4 = prev_close + 1.1 * (prev_high - prev_low) * 1.125 / 2
    camarilla_l4 = prev_close - 1.1 * (prev_high - prev_low) * 1.125 / 2
    
    # Calculate 1d HMA21 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    # Hull Moving Average: WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(values, window):
        if len(values) < window:
            return np.full_like(values, np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights / weights.sum(), mode='valid')
    
    # Pad arrays for WMA calculation
    wma_half = np.full_like(daily_close, np.nan)
    wma_full = np.full_like(daily_close, np.nan)
    
    if len(daily_close) >= half_len:
        wma_vals = wma(daily_close, half_len)
        wma_half[half_len-1:] = wma_vals
    
    if len(daily_close) >= 21:
        wma_vals = wma(daily_close, 21)
        wma_full[20:] = wma_vals
    
    raw_hma = 2 * wma_half - wma_full
    wma_final = np.full_like(raw_hma, np.nan)
    if len(raw_hma) >= sqrt_len:
        wma_vals = wma(raw_hma[~np.isnan(raw_hma)], sqrt_len)
        valid_idx = ~np.isnan(raw_hma)
        if np.sum(valid_idx) >= sqrt_len:
            wma_final[valid_idx] = np.nan
            wma_final[valid_idx][sqrt_len-1:] = wma_vals
    
    hma_21_1d = wma_final
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    # Trend filter: bullish if close > HMA21, bearish if close < HMA21
    bullish_regime = close > hma_21_1d_aligned
    bearish_regime = close < hma_21_1d_aligned
    
    # Volume confirmation: volume > 1.8 * 24-period average (strong spike)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(1, 24) + 1  # Need Camarilla (1 bar lag), volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or 
            np.isnan(hma_21_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above H4 AND bullish regime AND volume confirmation
            if close[i] > camarilla_h4[i] and bullish_regime[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below L4 AND bearish regime AND volume confirmation
            elif close[i] < camarilla_l4[i] and bearish_regime[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below L4 (opposite level)
            if close[i] < camarilla_l4[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above H4 (opposite level)
            if close[i] > camarilla_h4[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H4L4_1dHMA21_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0