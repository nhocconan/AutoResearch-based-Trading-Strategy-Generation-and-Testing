#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme with 1d EMA50 trend filter and volume confirmation
# Long when Williams %R < -80 (oversold) AND price > 1d EMA50 AND volume > 2.0x 24-bar avg
# Short when Williams %R > -20 (overbought) AND price < 1d EMA50 AND volume > 2.0x 24-bar avg
# Exit when Williams %R returns to -50 (mean reversion)
# Uses discrete position sizing (0.25) to reduce fee drag and improve test generalization.
# Target: 15-25 trades/year on 6h timeframe (60-100 total over 4 years) to avoid overtrading.
# Williams %R identifies exhaustion points in both bull and bear markets, while 1d EMA50 ensures
# alignment with the higher timeframe trend, preventing counter-trend trades. Volume confirmation
# filters out low-conviction moves. This combination works in ranging markets (mean reversion
# from extremes) and trending markets (pullbacks to EMA50 in direction of trend).

name = "6h_WilliamsRExtreme_1dEMA50_VolumeConfirm_v1"
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams %R calculation on 6h data (14-period)
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: >2.0x 24-bar average volume (stricter to reduce trades)
    volume_series = pd.Series(volume)
    volume_ma_24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > 2.0 * volume_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 50, period)  # volume MA, EMA50, and Williams %R warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_ma_24[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_ema50_1d = ema_50_1d_aligned[i]
        curr_williams_r = williams_r[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: Williams %R returns to -50 (mean reversion)
            if curr_williams_r >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R returns to -50 (mean reversion)
            if curr_williams_r <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when Williams %R < -80 (oversold) AND price > 1d EMA50 AND volume confirmation
            if curr_williams_r < -80 and curr_close > curr_ema50_1d and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when Williams %R > -20 (overbought) AND price < 1d EMA50 AND volume confirmation
            elif curr_williams_r > -20 and curr_close < curr_ema50_1d and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals