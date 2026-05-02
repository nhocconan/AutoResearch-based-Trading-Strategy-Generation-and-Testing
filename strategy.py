#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume confirmation
# Uses 1d timeframe for signal generation with Camarilla pivot breakouts from prior day
# 1w EMA(50) determines primary trend direction (bullish/bearish) - multi-timeframe alignment
# Volume confirmation (2.0x 20-period average) ensures institutional participation
# Discrete position sizing (0.25) balances return and risk while minimizing fee drag
# Target: 30-100 total trades over 4 years = 7-25/year for 1d timeframe
# Camarilla provides mathematical price levels based on prior day's range
# Volume confirms breakout validity, 1w EMA filter ensures trades only in higher timeframe trend direction
# Works in both bull and bear markets by only taking trades aligned with 1w trend

name = "1d_Camarilla_R3S3_Breakout_1wEMA50_Volume_Trend_v1"
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
    
    # Load 1w HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 1w EMA(50) for trend determination
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d Camarilla levels (based on prior day's high-low-close)
    # We need to shift by 1 to use prior day's levels (lookback)
    high_1d_shifted = np.roll(high, 1)
    low_1d_shifted = np.roll(low, 1)
    close_1d_shifted = np.roll(close, 1)
    # Set first value to NaN since no prior day exists
    high_1d_shifted[0] = np.nan
    low_1d_shifted[0] = np.nan
    close_1d_shifted[0] = np.nan
    
    hl_range = high_1d_shifted - low_1d_shifted
    camarilla_h3 = close_1d_shifted + 1.1 * hl_range / 4  # R3 resistance
    camarilla_l3 = close_1d_shifted - 1.1 * hl_range / 4  # S3 support
    
    # Align Camarilla levels to 1d timeframe (already aligned as we're on 1d)
    camarilla_h3_aligned = camarilla_h3  # No alignment needed for same timeframe
    camarilla_l3_aligned = camarilla_l3
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators and prior day)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(volume_confirm[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Close > Camarilla H3 (R3) + volume confirm + price > 1w EMA50 (bullish trend)
            if close[i] > camarilla_h3_aligned[i] and volume_confirm[i] and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close < Camarilla L3 (S3) + volume confirm + price < 1w EMA50 (bearish trend)
            elif close[i] < camarilla_l3_aligned[i] and volume_confirm[i] and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close < Camarilla L3 (S3) or price < 1w EMA50 (trend reversal)
            if close[i] < camarilla_l3_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close > Camarilla H3 (R3) or price > 1w EMA50 (trend reversal)
            if close[i] > camarilla_h3_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals