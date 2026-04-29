#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with 1w EMA trend filter and volume confirmation
# Long when price breaks above R3 AND price > 1w EMA50 AND volume > 2.0x 20-period average
# Short when price breaks below S3 AND price < 1w EMA50 AND volume > 2.0x 20-period average
# Uses discrete position sizing (0.30) to minimize fee drag and control drawdown.
# Target: 30-100 total trades over 4 years (7-25/year) to avoid fee drag kill zone.

name = "1d_Camarilla_R3S3_Breakout_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1w calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Camarilla pivot levels from previous day (using 1d data)
    # Pivot = (H + L + C) / 3
    # R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), R2 = C + ((H-L) * 1.1/6), R1 = C + ((H-L) * 1.1/12)
    # S1 = C - ((H-L) * 1.1/12), S2 = C - ((H-L) * 1.1/6), S3 = C - ((H-L) * 1.1/4), S4 = C - ((H-L) * 1.1/2)
    # We use R3/S3 for breakout signals
    pivot = (high + low + close) / 3
    range_hl = high - low
    r3 = close + (range_hl * 1.1 / 4)
    s3 = close - (range_hl * 1.1 / 4)
    
    # Shift pivot levels by 1 to use previous day's levels for today's breakout
    r3_prev = np.roll(r3, 1)
    s3_prev = np.roll(s3, 1)
    r3_prev[0] = np.nan  # First bar has no previous day
    s3_prev[0] = np.nan
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema50_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_r3 = r3_prev[i]  # Previous day's R3
        curr_s3 = s3_prev[i]  # Previous day's S3
        curr_ema50 = ema50_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation
            if curr_volume_confirm and not np.isnan(curr_r3) and not np.isnan(curr_s3):
                # Bullish entry: price breaks above R3 AND price > 1w EMA50
                if curr_close > curr_r3 and curr_close > curr_ema50:
                    signals[i] = 0.30
                    position = 1
                # Bearish entry: price breaks below S3 AND price < 1w EMA50
                elif curr_close < curr_s3 and curr_close < curr_ema50:
                    signals[i] = -0.30
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when price falls below previous day's pivot (mean reversion) OR trend changes
            pivot_prev = (np.roll(high, 1)[i] + np.roll(low, 1)[i] + np.roll(close, 1)[i]) / 3
            if i == 0:
                pivot_prev = np.nan
            if not np.isnan(pivot_prev) and curr_close < pivot_prev:
                signals[i] = 0.0
                position = 0
            elif curr_close < curr_ema50:  # Trend filter exit
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position - exit conditions
            # Exit when price rises above previous day's pivot OR trend changes
            pivot_prev = (np.roll(high, 1)[i] + np.roll(low, 1)[i] + np.roll(close, 1)[i]) / 3
            if i == 0:
                pivot_prev = np.nan
            if not np.isnan(pivot_prev) and curr_close > pivot_prev:
                signals[i] = 0.0
                position = 0
            elif curr_close > curr_ema50:  # Trend filter exit
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals