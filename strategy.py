#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme + 12h EMA50 Trend + Volume Spike
# Long when Williams %R(14) crosses above -20 (oversold bounce) AND price > 12h EMA50 AND volume > 1.8x 20-bar avg
# Short when Williams %R(14) crosses below -80 (overbought rejection) AND price < 12h EMA50 AND volume > 1.8x 20-bar avg
# Exit when Williams %R returns to -50 (mean reversion) or opposite signal triggers
# Williams %R identifies exhaustion points in ranging markets, EMA50 filters trend alignment,
# volume confirmation ensures momentum validity. Effective in both bull/bear regimes via mean reversion.

name = "6h_WilliamsRExtreme_12hEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 6h data for Williams %R calculation (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 14:
        return np.zeros(n)
    
    # Calculate Williams %R(14) on 6h data
    highest_high = pd.Series(df_6h['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_6h['low'].values).rolling(window=14, min_periods=14).min().values
    close_6h = df_6h['close'].values
    williams_r = -100 * (highest_high - close_6h) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to original timeframe (should be identical since 6h->6h)
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h data
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 6h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # volume MA and EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_williams_r = williams_r_aligned[i]
        curr_ema50 = ema_50_12h_aligned[i]
        curr_close = close[i]
        prev_williams_r = williams_r_aligned[i-1] if i > 0 else -50
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: Williams %R returns to -50 (mean reversion) or bearish reversal
            if curr_williams_r >= -50 or (curr_williams_r < -80 and curr_close < curr_ema50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R returns to -50 (mean reversion) or bullish reversal
            if curr_williams_r <= -50 or (curr_williams_r > -20 and curr_close > curr_ema50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when Williams %R crosses above -20 (oversold bounce) AND price > EMA50 AND volume confirmation
            if prev_williams_r <= -20 and curr_williams_r > -20 and curr_close > curr_ema50 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when Williams %R crosses below -80 (overbought rejection) AND price < EMA50 AND volume confirmation
            elif prev_williams_r >= -80 and curr_williams_r < -80 and curr_close < curr_ema50 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals