#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Camarilla Pivot + 1D Trend + Volume
# Hypothesis: Camarilla pivot levels from 1-day provide precise support/resistance.
# Fade at R3/S3 (85% retracement) with trend filter from 1-day EMA(50).
# Breakout continuation at R4/S4 (100%+ extension) with volume confirmation.
# Works in bull/bear: mean reversion in range, trend following in breakout.
# Target: 20-40 trades/year to minimize fee drag on 6h timeframe.
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
    
    # Get 1-day data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 1-day bar
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    # where C, H, L are from previous day (to avoid look-ahead)
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    # Shift by 1 to use previous day's data only
    prev_close = np.roll(daily_close, 1)
    prev_high = np.roll(daily_high, 1)
    prev_low = np.roll(daily_low, 1)
    prev_close[0] = np.nan  # First value invalid after roll
    
    # Calculate pivot levels
    r4 = prev_close + ((prev_high - prev_low) * 1.1 / 2)
    r3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)
    s3 = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    s4 = prev_close - ((prev_high - prev_low) * 1.1 / 2)
    
    # Align to 6h timeframe
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # 1-day EMA(50) for trend filter
    daily_ema = pd.Series(daily_close).ewm(span=50, adjust=False).mean().values
    daily_ema_6h = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(r4_6h[i]) or np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(s4_6h[i]) or np.isnan(daily_ema_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price below S3 (fade failed) or trend turns bearish
            if close[i] < s3_6h[i] or close[i] < daily_ema_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price above R3 (fade failed) or trend turns bullish
            if close[i] > r3_6h[i] or close[i] > daily_ema_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Fade at R3/S3 with volume confirmation
            if vol_filter[i]:
                # Long fade at S3: price <= S3 and above S4 (within zone) + bullish trend
                if close[i] <= s3_6h[i] and close[i] >= s4_6h[i] and close[i] > daily_ema_6h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short fade at R3: price >= R3 and below R4 (within zone) + bearish trend
                elif close[i] >= r3_6h[i] and close[i] <= r4_6h[i] and close[i] < daily_ema_6h[i]:
                    position = -1
                    signals[i] = -0.25
            # Breakout continuation: strong volume + break beyond R4/S4
            elif volume[i] > (vol_ma[i] * 2.0):  # Strong volume breakout
                # Long breakout: price > R4 + bearish to bullish trend shift
                if close[i] > r4_6h[i] and daily_ema_6h[i] > np.roll(daily_ema_6h, 1)[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price < S4 + bullish to bearish trend shift
                elif close[i] < s4_6h[i] and daily_ema_6h[i] < np.roll(daily_ema_6h, 1)[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals