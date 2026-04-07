#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Camarilla Pivot + Volume Confirmation + Daily Trend Filter
# Hypothesis: Camarilla pivot levels from daily chart provide strong support/resistance.
# Long at S1/S2 with bullish daily trend and volume confirmation; short at R1/R2 with bearish daily trend.
# Works in both bull and bear markets by fading extremes in ranging conditions and
# following breakouts in trending markets. Target: 20-30 trades/year.
name = "6h_camarilla_pivot_1d_volume_trend_v2"
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
    
    # Daily OHLC for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla multipliers
    # R4 = Close + (High - Low) * 1.1/2
    # R3 = Close + (High - Low) * 1.1/4
    # R2 = Close + (High - Low) * 1.1/6
    # R1 = Close + (High - Low) * 1.1/12
    # S1 = Close - (High - Low) * 1.1/12
    # S2 = Close - (High - Low) * 1.1/6
    # S3 = Close - (High - Low) * 1.1/4
    # S4 = Close - (High - Low) * 1.1/2
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_r2 = prev_close + (prev_high - prev_low) * 1.1 / 6
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    camarilla_s2 = prev_close - (prev_high - prev_low) * 1.1 / 6
    
    # Align to 6h timeframe
    r1_6h = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    r2_6h = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    s1_6h = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    s2_6h = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    
    # Daily EMA(50) for trend filter
    daily_close = df_1d['close'].values
    daily_ema = pd.Series(daily_close).ewm(span=50, adjust=False).mean().values
    daily_ema_6h = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    # Volume ratio (current vs 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(r1_6h[i]) or np.isnan(r2_6h[i]) or 
            np.isnan(s1_6h[i]) or np.isnan(s2_6h[i]) or
            np.isnan(daily_ema_6h[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below S2 or daily trend turns bearish
            if close[i] < s2_6h[i] or close[i] < daily_ema_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price crosses above R2 or daily trend turns bullish
            if close[i] > r2_6h[i] or close[i] > daily_ema_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation (vol > 1.2x average)
            if vol_ratio[i] > 1.2:
                # Enter long: price above S1 but below S2, bullish daily trend
                if s1_6h[i] < close[i] < s2_6h[i] and close[i] > daily_ema_6h[i]:
                    position = 1
                    signals[i] = 0.25
                # Enter short: price below R1 but above R2, bearish daily trend
                elif r2_6h[i] < close[i] < r1_6h[i] and close[i] < daily_ema_6h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals