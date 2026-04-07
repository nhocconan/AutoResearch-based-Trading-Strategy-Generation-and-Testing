#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Camarilla Pivot + 1D Trend + Volume Confirmation
# Hypothesis: Camarilla pivot levels on 4h provide strong support/resistance.
# We trade reversals at S3/R3 levels when 1-day EMA(50) confirms trend direction,
# with volume spike confirmation. This captures mean reversion in trends with
# institutional levels. Works in bull/bear by following higher timeframe trend.
# Target: 20-50 trades/year to minimize fee drag on 4h timeframe.
name = "4h_camarilla_pivot_1d_trend_volume_v1"
timeframe = "4h"
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
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for 4h timeframe
    # Using previous bar's high, low, close
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    
    # First bar has no previous data
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Pivot point and range
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Camarilla levels: S3, S2, S1, R1, R2, R3
    # S3 = Close - (High - Low) * 1.1000 / 4
    # S2 = Close - (High - Low) * 1.1000 / 6
    # S1 = Close - (High - Low) * 1.1000 / 12
    # R1 = Close + (High - Low) * 1.1000 / 12
    # R2 = Close + (High - Low) * 1.1000 / 6
    # R3 = Close + (High - Low) * 1.1000 / 4
    s3 = prev_close - (range_val * 1.1000 / 4)
    s2 = prev_close - (range_val * 1.1000 / 6)
    s1 = prev_close - (range_val * 1.1000 / 12)
    r1 = prev_close + (range_val * 1.1000 / 12)
    r2 = prev_close + (range_val * 1.1000 / 6)
    r3 = prev_close + (range_val * 1.1000 / 4)
    
    # 1-day EMA(50) for trend filter
    daily_close = df_1d['close'].values
    daily_ema = pd.Series(daily_close).ewm(span=50, adjust=False).mean().values
    daily_ema_4h = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    # Volume filter: current volume > 2.0x 20-period average (strict for fewer trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(s3[i]) or np.isnan(r3[i]) or np.isnan(daily_ema_4h[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches R3 (take profit) or trend turns bearish
            if close[i] >= r3[i] or close[i] < daily_ema_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price reaches S3 (take profit) or trend turns bullish
            if close[i] <= s3[i] or close[i] > daily_ema_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation
            if vol_filter[i]:
                # Long at S3: price <= S3 + 0.1% buffer + bullish trend
                if close[i] <= s3[i] * 1.001 and close[i] > daily_ema_4h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short at R3: price >= R3 - 0.1% buffer + bearish trend
                elif close[i] >= r3[i] * 0.999 and close[i] < daily_ema_4h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals