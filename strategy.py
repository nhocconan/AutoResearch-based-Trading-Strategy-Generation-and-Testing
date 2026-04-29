#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot R3/S3 breakout with 4h EMA50 trend filter and volume spike
# Long when price breaks above R3 AND price > 4h EMA50 AND volume > 2x 20-period average
# Short when price breaks below S3 AND price < 4h EMA50 AND volume > 2x 20-period average
# Exit when price reverts to opposite pivot level (R2 for longs, S2 for shorts) or volume dries up
# Uses 4h/1d for signal direction, 1h only for entry timing precision
# Target: 60-150 total trades over 4 years = 15-37/year for 1h
# Session filter: 08-20 UTC to reduce noise trades
# Position size: 0.20 (discrete level to minimize fee churn)

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load HTF data ONCE before loop for 4h calculations
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA50 to 1h timeframe (completed 4h bar only)
    ema50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate daily pivot points for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for today's Camarilla levels
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla levels: R3/S3 and R2/S2 for exits
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    # R2 = close + (high - low) * 1.1/6
    # S2 = close - (high - low) * 1.1/6
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    camarilla_r2 = prev_close + (prev_high - prev_low) * 1.1 / 6
    camarilla_s2 = prev_close - (prev_high - prev_low) * 1.1 / 6
    
    # Align daily Camarilla levels to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 20)  # warmup for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0  # force close at session end
                position = 0
            else:
                signals[i] = 0.0
            continue
            
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
        curr_r2 = r2_aligned[i]
        curr_s2 = s2_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Trend regime: bullish if price > 4h EMA50, bearish if price < 4h EMA50
        is_bullish_regime = curr_close > curr_ema50
        is_bearish_regime = curr_close < curr_ema50
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and in session
            if curr_volume_confirm:
                # Bullish entry: price breaks above R3 AND bullish regime
                if curr_high > curr_r3 and is_bullish_regime:
                    signals[i] = 0.20
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below S3 AND bearish regime
                elif curr_low < curr_s3 and is_bearish_regime:
                    signals[i] = -0.20
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price reverts to R2 OR volume dries up OR regime changes to bearish
            if curr_close < curr_r2 or not curr_volume_confirm or not is_bullish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price reverts to S2 OR volume dries up OR regime changes to bullish
            if curr_close > curr_s2 or not curr_volume_confirm or not is_bearish_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals