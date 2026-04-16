#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1w pivot direction and volume confirmation.
# Long when price breaks above 1d Camarilla R3 AND 1w pivot > prior 1w pivot AND volume > 1.5x 20-period average.
# Short when price breaks below 1d Camarilla S3 AND 1w pivot < prior 1w pivot AND volume > 1.5x 20-period average.
# Exit on opposite Camarilla break (R3/S3) or ATR(14) stoploss (2*ATR from entry).
# Uses discrete position size 0.25. Designed to capture strong intraday momentum with institutional pivot confirmation.
# Weekly pivot filter ensures alignment with major trend, reducing false breakouts in ranging markets.
# Volume confirmation avoids low-liquidity false signals.
# Target: 75-150 total trades over 4 years (19-38/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Indicators: Camarilla Pivots (R3, S3) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate pivots from previous 1d bar
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    pivot_1d = (prev_high + prev_low + prev_close) / 3.0
    range_1d = prev_high - prev_low
    r3_1d = pivot_1d + range_1d * 1.1 / 4.0  # R3 = pivot + (high-low)*1.1/4
    s3_1d = pivot_1d - range_1d * 1.1 / 4.0  # S3 = pivot - (high-low)*1.1/4
    
    # Align to 6h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # === 1w Indicators: Weekly Pivot for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly pivot from previous week
    prev_close_w = np.roll(close_1w, 1)
    prev_high_w = np.roll(high_1w, 1)
    prev_low_w = np.roll(low_1w, 1)
    pivot_1w = (prev_high_w + prev_low_w + prev_close_w) / 3.0
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    pivot_1w_prev = np.roll(pivot_1w_aligned, 1)
    pivot_up = pivot_1w_aligned > pivot_1w_prev   # Weekly pivot rising
    pivot_down = pivot_1w_aligned < pivot_1w_prev  # Weekly pivot falling
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    # === 6h ATR for stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_6h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_6h_raw = pd.Series(tr_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or np.isnan(pivot_1w_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(atr_6h_raw[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        atr_val = atr_6h_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below 1d Camarilla S3
            if price < s3_1d_aligned[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above 1d Camarilla R3
            if price > r3_1d_aligned[i]:
                exit_signal = True
            # ATR-based stoploss: 2*ATR above entry
            elif price > entry_price + 2.0 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above 1d Camarilla R3 AND weekly pivot rising AND volume spike
            if price > r3_1d_aligned[i] and pivot_up[i] and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below 1d Camarilla S3 AND weekly pivot falling AND volume spike
            elif price < s3_1d_aligned[i] and pivot_down[i] and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_Camarilla_R3_S3_1wPivot_VolumeSpike_V1"
timeframe = "6h"
leverage = 1.0