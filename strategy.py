#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Camarilla Pivot + 1d Trend + Volume Confirmation
# Hypothesis: Camarilla pivot levels from daily data provide strong support/resistance.
# In bull regime (1d EMA > SMA), we go long on bounce from S3/S4 with volume.
# In bear regime (1d EMA < SMA), we go short on rejection at R3/R4 with volume.
# Uses 6h timeframe for execution with daily pivot levels for structure.
# Target: 15-35 trades/year (60-140 over 4 years) to avoid fee drag.
name = "6h_camarilla_pivot_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    # Using previous day's OHLC (standard Camarilla calculation)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Pivot point = (H + L + C) / 3
    pivot = (daily_high + daily_low + daily_close) / 3.0
    # Range = H - L
    range_val = daily_high - daily_low
    
    # Camarilla levels
    # Resistance levels
    r1 = pivot + (range_val * 1.1 / 12)
    r2 = pivot + (range_val * 1.1 / 6)
    r3 = pivot + (range_val * 1.1 / 4)
    r4 = pivot + (range_val * 1.1 / 2)
    # Support levels
    s1 = pivot - (range_val * 1.1 / 12)
    s2 = pivot - (range_val * 1.1 / 6)
    s3 = pivot - (range_val * 1.1 / 4)
    s4 = pivot - (range_val * 1.1 / 2)
    
    # Align daily levels to 6h timeframe (shifted by 1 day for look-ahead bias)
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # 1-day trend filter: EMA(20) > SMA(20) = bullish, else bearish
    daily_ema20 = pd.Series(daily_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    daily_sma20 = pd.Series(daily_close).rolling(window=20, min_periods=20).mean().values
    daily_ema20_6h = align_htf_to_ltf(prices, df_1d, daily_ema20)
    daily_sma20_6h = align_htf_to_ltf(prices, df_1d, daily_sma20)
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(pivot_6h[i]) or np.isnan(r3_6h[i]) or np.isnan(r4_6h[i]) or 
            np.isnan(s3_6h[i]) or np.isnan(s4_6h[i]) or np.isnan(daily_ema20_6h[i]) or 
            np.isnan(daily_sma20_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend regime from 1d data
        bull_regime = daily_ema20_6h[i] > daily_sma20_6h[i]  # Bullish when EMA > SMA
        bear_regime = daily_ema20_6h[i] < daily_sma20_6h[i]  # Bearish when EMA < SMA
        
        if position == 1:  # Long position
            # Exit: price breaks below S3 with volume or trend turns bearish
            if close[i] < s3_6h[i] and vol_filter[i]:
                position = 0
                signals[i] = 0.0
            elif not bull_regime:  # Trend turned bearish
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price breaks above R3 with volume or trend turns bullish
            if close[i] > r3_6h[i] and vol_filter[i]:
                position = 0
                signals[i] = 0.0
            elif not bear_regime:  # Trend turned bullish
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation
            if vol_filter[i]:
                # Bull regime: look for long when price bounces from S3/S4
                if bull_regime:
                    # Long signal: price touches or goes below S3/S4 then closes back above
                    if i > 50:
                        was_below_s3 = low[i-1] <= s3_6h[i-1]
                        was_below_s4 = low[i-1] <= s4_6h[i-1]
                        now_above_s3 = close[i] > s3_6h[i]
                        now_above_s4 = close[i] > s4_6h[i]
                        if ((was_below_s3 or was_below_s4) and (now_above_s3 or now_above_s4)):
                            position = 1
                            signals[i] = 0.25
                # Bear regime: look for short when price rejects R3/R4
                elif bear_regime:
                    # Short signal: price touches or goes above R3/R4 then closes back below
                    if i > 50:
                        was_above_r3 = high[i-1] >= r3_6h[i-1]
                        was_above_r4 = high[i-1] >= r4_6h[i-1]
                        now_below_r3 = close[i] < r3_6h[i]
                        now_below_r4 = close[i] < r4_6h[i]
                        if ((was_above_r3 or was_above_r4) and (now_below_r3 or now_below_r4)):
                            position = -1
                            signals[i] = -0.25
    
    return signals