#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h 12h/1d Camarilla pivot levels with volume confirmation and ATR filter.
# Fade at R3/S3 levels (mean reversion in ranging markets), breakout continuation at R4/S4 (trend continuation).
# Uses 12h EMA50 trend filter to align with higher timeframe direction.
# Designed for 6h timeframe to capture both mean reversion and trend continuation with low trade frequency.
# Entry: Long at S3 bounce with 12h EMA50 uptrend + volume spike, Short at R3 rejection with 12h EMA50 downtrend + volume spike.
# Exit: Opposite pivot level touch or trend filter failure.
# Uses strict conditions to limit trades (~15-25/year) and avoid overtrading.
name = "6h_12h1d_Camarilla_Pivot_Volume_ATR"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for pivot calculation and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 12h bar
    # Typical price = (H + L + C) / 3
    typical_price = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3
    # Pivot point
    pivot = typical_price
    # R1, S1
    r1 = pivot * 2 - df_12h['low']
    s1 = pivot * 2 - df_12h['high']
    # R2, S2
    r2 = pivot + (df_12h['high'] - df_12h['low'])
    s2 = pivot - (df_12h['high'] - df_12h['low'])
    # R3, S3
    r3 = pivot + (df_12h['high'] - df_12h['low']) * 1.1
    s3 = pivot - (df_12h['high'] - df_12h['low']) * 1.1
    # R4, S4
    r4 = pivot + (df_12h['high'] - df_12h['low']) * 1.5
    s4 = pivot - (df_12h['high'] - df_12h['low']) * 1.5
    
    # Align pivot levels to 6h timeframe (using previous bar's values)
    pivot_12h = align_htf_to_ltf(prices, df_12h, pivot.values)
    r1_12h = align_htf_to_ltf(prices, df_12h, r1.values)
    s1_12h = align_htf_to_ltf(prices, df_12h, s1.values)
    r2_12h = align_htf_to_ltf(prices, df_12h, r2.values)
    s2_12h = align_htf_to_ltf(prices, df_12h, s2.values)
    r3_12h = align_htf_to_ltf(prices, df_12h, r3.values)
    s3_12h = align_htf_to_ltf(prices, df_12h, s3.values)
    r4_12h = align_htf_to_ltf(prices, df_12h, r4.values)
    s4_12h = align_htf_to_ltf(prices, df_12h, s4.values)
    
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # ATR for volatility filter (6-period ATR)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=6, min_periods=6).mean().values
    
    # Volume filter: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_12h[i]) or np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or
            np.isnan(r4_12h[i]) or np.isnan(s4_12h[i]) or np.isnan(ema50_12h_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price at S3 level (mean reversion) with uptrend and volume spike
            if (low[i] <= s3_12h[i] * 1.001 and  # Allow small tolerance for touching S3
                close[i] > s3_12h[i] and          # Price bounces above S3
                close[i] > ema50_12h_aligned[i] and  # Above 12h EMA50 (uptrend)
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price at R3 level (mean reversion) with downtrend and volume spike
            elif (high[i] >= r3_12h[i] * 0.999 and   # Allow small tolerance for touching R3
                  close[i] < r3_12h[i] and           # Price rejects below R3
                  close[i] < ema50_12h_aligned[i] and  # Below 12h EMA50 (downtrend)
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            # Long breakout: Price breaks above R4 with uptrend and volume spike
            elif (close[i] > r4_12h[i] and
                  close[i] > ema50_12h_aligned[i] and
                  volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short breakout: Price breaks below S4 with downtrend and volume spike
            elif (close[i] < s4_12h[i] and
                  close[i] < ema50_12h_aligned[i] and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price reaches R3 (mean reversion target) or breaks below S4 (stop)
            # or if trend turns down
            if (high[i] >= r3_12h[i] * 0.999 or  # Hit R3 target
                close[i] < s4_12h[i] or          # Break below S4 (stop)
                close[i] < ema50_12h_aligned[i]): # Trend turned down
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price reaches S3 (mean reversion target) or breaks above R4 (stop)
            # or if trend turns up
            if (low[i] <= s3_12h[i] * 1.001 or   # Hit S3 target
                close[i] > r4_12h[i] or          # Break above R4 (stop)
                close[i] > ema50_12h_aligned[i]): # Trend turned up
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals