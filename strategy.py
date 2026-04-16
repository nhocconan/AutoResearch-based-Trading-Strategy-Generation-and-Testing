#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h trend filter (HMA21), volume confirmation (>1.5x 20-bar avg), and session filter (08-20 UTC).
# Long when price breaks above Camarilla R1 AND 4h HMA21 trending up AND volume > 1.5x 20-bar average.
# Short when price breaks below Camarilla S1 AND 4h HMA21 trending down AND volume > 1.5x 20-bar average.
# Exit on opposite Camarilla break (S1 for long, R1 for short) or time-based exit (max 12 bars held).
# Uses discrete position size 0.20. Designed to capture intraday momentum with HTF trend alignment and volume confirmation.
# Target: 60-150 total trades over 4 years (15-37/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1h Indicators: Camarilla Pivot Levels (based on previous bar) ===
    # Camarilla: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    # Use previous bar's high/low/close to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = prev_high[1] if n > 1 else high[0]  # handle first bar
    prev_low[0] = prev_low[1] if n > 1 else low[0]
    prev_close[0] = prev_close[1] if n > 1 else close[0]
    
    camarilla_upper = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_lower = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # === 4h Indicators: HMA(21) for trend ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    # HMA calculation: WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    half_len = 10  # 21/2 = 10.5 -> 10
    sqrt_len = 4   # sqrt(21) ≈ 4.58 -> 4
    wma_half = pd.Series(close_4h).ewm(span=half_len, adjust=False, min_periods=half_len).mean().values
    wma_full = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    raw_hma = 2 * wma_half - wma_full
    hma_4h = pd.Series(raw_hma).ewm(span=sqrt_len, adjust=False, min_periods=sqrt_len).mean().values
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    hma_up = hma_4h_aligned > np.roll(hma_4h_aligned, 1)
    hma_down = hma_4h_aligned < np.roll(hma_4h_aligned, 1)
    
    # === 1h Indicators: Volume Spike (>1.5x 20-bar average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 40 periods needed)
    warmup = 40
    
    # Track position state and entry price
    position = 0  # 0: flat, 1: long, -1: short
    bars_held = 0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(camarilla_upper[i]) or np.isnan(camarilla_lower[i]) or np.isnan(hma_4h_aligned[i]) or
            np.isnan(volume_spike[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            bars_held = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below Camarilla S1
            if price < camarilla_lower[i]:
                exit_signal = True
            # Time-based exit: max 12 bars held
            elif bars_held >= 12:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above Camarilla R1
            if price > camarilla_upper[i]:
                exit_signal = True
            # Time-based exit: max 12 bars held
            elif bars_held >= 12:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            bars_held = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Camarilla R1 AND HMA trending up AND volume spike
            if price > camarilla_upper[i] and hma_up[i] and vol_spike:
                signals[i] = 0.20
                position = 1
                bars_held = 1
            
            # SHORT: Price breaks below Camarilla S1 AND HMA trending down AND volume spike
            elif price < camarilla_lower[i] and hma_down[i] and vol_spike:
                signals[i] = -0.20
                position = -1
                bars_held = 1
        
        else:
            signals[i] = position * 0.20
            bars_held += 1
    
    return signals

name = "1h_Camarilla_R1S1_4hHMA21_VolumeSpike_V1"
timeframe = "1h"
leverage = 1.0