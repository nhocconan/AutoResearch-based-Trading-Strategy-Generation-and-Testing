#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R extreme with 1d EMA50 trend filter and volume confirmation
# Long when Williams %R < -80 (oversold) AND price > 1d EMA50 AND volume > 1.5x 20-bar avg
# Short when Williams %R > -20 (overbought) AND price < 1d EMA50 AND volume > 1.5x 20-bar avg
# Exit when Williams %R crosses back above -50 (long) or below -50 (short)
# Uses discrete position sizing (0.25) to reduce fee drag while maintaining profitability.
# Williams %R captures exhaustion points; 1d EMA50 filters counter-trend moves in bear markets.
# Volume confirmation ensures institutional participation, reducing false signals.
# Target: 30-60 trades/year on 4h (120-240 total over 4 years) to avoid overtrading.

name = "4h_WilliamsR_Extreme_1dEMA50_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA(50) on 1d data
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R(14) on 4h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14, 20)  # EMA50, Williams %R, and volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        ema_50 = ema_50_1d_aligned[i]
        wr = williams_r[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: Williams %R crosses back above -50 (exhaustion unwinding)
            if wr > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R crosses back below -50 (exhaustion unwinding)
            if wr < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when Williams %R < -80 (oversold) AND price > 1d EMA50 AND volume confirmation
            if wr < -80 and curr_close > ema_50 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when Williams %R > -20 (overbought) AND price < 1d EMA50 AND volume confirmation
            elif wr > -20 and curr_close < ema_50 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals