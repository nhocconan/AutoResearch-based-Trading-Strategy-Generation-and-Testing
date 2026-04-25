#!/usr/bin/env python3
"""
6h_Camarilla_R4S4_Breakout_12hTrend_VolumeSpike
Hypothesis: 6h Camarilla R4/S4 breakout with 12h EMA50 trend filter and volume spike confirmation.
Long when price breaks above R4 in 12h uptrend (close > 12h EMA50) with volume > 2.0x 20-period average.
Short when price breaks below S4 in 12h downtrend (close < 12h EMA50) with volume > 2.0x 20-period average.
Exit when price re-enters the Camarilla H3-L3 range (mean reversion zone).
Designed for ~12-25 trades/year by requiring strong breakouts and volume confirmation.
Works in bull/bear markets via 12h EMA50 filter; avoids whipsaws via volume confirmation.
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
    
    # Get 12h data for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla levels from previous day
    # Using daily OHLC from 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC (shifted by 1 to avoid look-ahead)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_open = df_1d['open'].shift(1).values
    
    # Align to 6h timeframe
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    prev_open_aligned = align_htf_to_ltf(prices, df_1d, prev_open)
    
    # Calculate Camarilla levels for previous day
    # R4 = Close + 1.5 * (High - Low)
    # S4 = Close - 1.5 * (High - Low)
    # H3 = Close + 1.125 * (High - Low)
    # L3 = Close - 1.125 * (High - Low)
    # H3 and L3 used for mean reversion exit
    range_hl = prev_high_aligned - prev_low_aligned
    r4 = prev_close_aligned + 1.5 * range_hl
    s4 = prev_close_aligned - 1.5 * range_hl
    h3 = prev_close_aligned + 1.125 * range_hl
    l3 = prev_close_aligned - 1.125 * range_hl
    
    # Volume regime: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_regime = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = max(100, 20)  # 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(r4[i]) or np.isnan(s4[i]) or 
            np.isnan(h3[i]) or np.isnan(l3[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend = ema_50_12h_aligned[i]
        
        if position == 0:
            # Only trade in trending regimes (12h EMA50 filter)
            if close[i] > ema_trend:  # 12h uptrend regime
                # Long: break above R4 with volume spike
                long_signal = (close[i] > r4[i]) and vol_regime[i]
            else:  # 12h downtrend regime
                # Short: break below S4 with volume spike
                short_signal = (close[i] < s4[i]) and vol_regime[i]
            
            if 'long_signal' in locals() and long_signal:
                signals[i] = 0.25
                position = 1
            elif 'short_signal' in locals() and short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
                # Clear signal variables for next iteration
                if 'long_signal' in locals(): del long_signal
                if 'short_signal' in locals(): del short_signal
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price re-enters H3-L3 range (mean reversion)
            if close[i] <= h3[i] and close[i] >= l3[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price re-enters H3-L3 range (mean reversion)
            if close[i] >= l3[i] and close[i] <= h3[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R4S4_Breakout_12hTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0