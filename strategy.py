#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h EMA50 trend filter and volume spike
# Long when price breaks above Camarilla R3 AND 4h EMA50 uptrend AND volume spike
# Short when price breaks below Camarilla S3 AND 4h EMA50 downtrend AND volume spike
# Exit on opposite Camarilla level (S1 for longs, R1 for shorts) or trend reversal
# Uses 4h/1d for signal direction, 1h only for entry timing precision
# Session filter 08-20 UTC to reduce noise trades
# Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag

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
    open_time = prices['open_time'].values
    
    # Session filter: 08-20 UTC (pre-compute before loop)
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
    
    # Calculate Camarilla pivots from previous day
    # Need daily high/low/close for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R4 = close + 1.5 * (high - low)
    # R3 = close + 1.1 * (high - low)
    # R2 = close + 0.55 * (high - low)
    # R1 = close + 0.275 * (high - low)
    # PP = (high + low + close) / 3
    # S1 = close - 0.275 * (high - low)
    # S2 = close - 0.55 * (high - low)
    # S3 = close - 1.1 * (high - low)
    # S4 = close - 1.5 * (high - low)
    
    # Calculate daily ranges
    daily_range = high_1d - low_1d
    
    # Camarilla levels based on previous day
    camarilla_r3 = close_1d + 1.1 * daily_range  # R3
    camarilla_s3 = close_1d - 1.1 * daily_range  # S3
    camarilla_r1 = close_1d + 0.275 * daily_range  # R1
    camarilla_s1 = close_1d - 0.275 * daily_range  # S1
    
    # Align Camarilla levels to 1h timeframe (use previous day's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: volume > 2.0x 24-period average (24*1h = 1 day)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (2.0 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24)  # warmup for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
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
        curr_r1 = r1_aligned[i]
        curr_s1 = s1_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Trend regime: bullish if price > 4h EMA50, bearish if price < 4h EMA50
        is_bullish_trend = curr_close > curr_ema50
        is_bearish_trend = curr_close < curr_ema50
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and in session
            if curr_volume_confirm:
                # Bullish entry: price breaks above R3 AND bullish 4h trend
                if curr_high > curr_r3 and is_bullish_trend:
                    signals[i] = 0.20
                    position = 1
                # Bearish entry: price breaks below S3 AND bearish 4h trend
                elif curr_low < curr_s3 and is_bearish_trend:
                    signals[i] = -0.20
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price breaks below S1 OR trend turns bearish
            if curr_low < curr_s1 or not is_bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price breaks above R1 OR trend turns bullish
            if curr_high > curr_r1 or not is_bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals