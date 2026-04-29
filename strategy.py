#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike
# Uses 4h primary timeframe with 12h HTF for trend and pivot calculation
# Camarilla R3/S3 breakouts indicate strong momentum when aligned with 12h trend
# Volume confirmation filters false breakouts
# Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag
# Works in both bull and bear markets by following 12h trend while capturing 4h momentum

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h timeframe (completed 12h bar only)
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla pivot levels from prior 12h (using 12h data)
    # Prior 12h period's OHLC for Camarilla calculation
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla levels: R3/S3 are strongest breakout levels
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    camarilla_range = (high_12h - low_12h) * 1.1 / 4
    r3_level = close_12h + camarilla_range
    s3_level = close_12h - camarilla_range
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3_level)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3_level)
    
    # Volume confirmation: volume > 2.0x 20-period average (20*4h = ~3.3 days)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema50_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema50 = ema50_aligned[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Trend regime: bullish if price > 12h EMA50, bearish if price < 12h EMA50
        is_bullish_regime = curr_close > curr_ema50
        is_bearish_regime = curr_close < curr_ema50
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation
            if curr_volume_confirm:
                # Bullish entry: break above R3 AND bullish regime
                if curr_high > curr_r3 and is_bullish_regime:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: break below S3 AND bearish regime
                elif curr_low < curr_s3 and is_bearish_regime:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price breaks below S3 (reversal) OR regime changes to bearish
            if curr_low < curr_s3 or not is_bullish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price breaks above R3 (reversal) OR regime changes to bullish
            if curr_high > curr_r3 or not is_bearish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals