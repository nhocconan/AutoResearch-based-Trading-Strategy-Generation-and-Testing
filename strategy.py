#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike
# Long when price breaks above R3 in bullish regime (price > 12h EMA50) with volume confirmation
# Short when price breaks below S3 in bearish regime (price < 12h EMA50) with volume confirmation
# Uses 12h EMA50 for trend filter to avoid whipsaws in counter-trend conditions
# Volume confirmation ensures breakouts have institutional participation
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
# Works in both bull and bear markets by only trading with the 12h trend

name = "6h_Camarilla_R3S3_Breakout_12hEMA50_VolumeSpike_HT"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 12h calculations
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align 12h EMA50 to 6h timeframe (completed 12h bar only)
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla pivot levels from daily OHLC
    # We need daily data for pivot calculation - resample from 6h to get proper daily OHLC
    # But since we can't resample, we'll use the 12h data to approximate daily
    # Better approach: use the actual 1d data from mtf_data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R4 = close + 1.5 * (high - low)
    # R3 = close + 1.0 * (high - low)
    # S3 = close - 1.0 * (high - low)
    # S4 = close - 1.5 * (high - low)
    rng = high_1d - low_1d
    r3 = close_1d + 1.0 * rng
    s3 = close_1d - 1.0 * rng
    
    # Align daily Camarilla levels to 6h timeframe (completed 1d bar only)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_50_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_ema50 = ema_50_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Trend filter: bullish if price > 12h EMA50, bearish if price < 12h EMA50
        is_bullish = curr_close > curr_ema50
        is_bearish = curr_close < curr_ema50
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation
            if curr_volume_confirm:
                # Bullish breakout: price breaks above R3 in bullish regime
                if is_bullish and curr_close > curr_r3:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below S3 in bearish regime
                elif is_bearish and curr_close < curr_s3:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when price returns to midpoint between R3 and S3 OR breaks below S3 with volume
            midpoint = (curr_r3 + curr_s3) / 2.0
            
            if curr_close <= midpoint or (curr_close < curr_s3 and curr_volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when price returns to midpoint between R3 and S3 OR breaks above R3 with volume
            midpoint = (curr_r3 + curr_s3) / 2.0
            
            if curr_close >= midpoint or (curr_close > curr_r3 and curr_volume_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals