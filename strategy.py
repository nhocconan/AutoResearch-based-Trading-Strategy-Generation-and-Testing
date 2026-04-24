#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator with 1d EMA200 trend filter and ATR volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for EMA200 trend filter and ATR-based volume confirmation.
- Williams Alligator: Jaw (EMA13 of median price, smoothed 8), Teeth (EMA8 of median price, smoothed 5), Lips (EMA5 of median price, smoothed 3).
- Entry: Long when Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA200 AND ATR(24) > 1.5 * ATR(100) (volume spike proxy).
         Short when Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA200 AND ATR(24) > 1.5 * ATR(100).
- Exit: Opposite Alligator alignment OR price crosses 1d EMA200 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Alligator identifies trend absence (all lines intertwined) and trend formation (lines diverge in bullish/bearish order).
- 1d EMA200 provides strong long-term trend filter to avoid counter-trend trades in bear markets.
- ATR ratio (short/long) serves as volume spike proxy since true volume data isn't available in HTF alignment for ATR.
- Works in bull markets (buy alignment in uptrend) and bear markets (sell alignment in downtrend).
- Estimated trades: ~120 total over 4 years (~30/year) based on Alligator alignment frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate median price for Alligator
    median_price = (high + low) / 2.0
    
    # Calculate 1d trend filter: EMA200
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 210:  # Need sufficient data for EMA200
        return np.zeros(n)
    
    ema200_1d = ema(df_1d['close'].values, 200)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d, additional_delay_bars=1)
    
    # Calculate 1d ATR for volume spike proxy (using ATR ratio)
    if len(df_1d) < 101:  # Need for ATR(100)
        return np.zeros(n)
    
    # True Range
    tr1 = df_1d['high'].values - df_1d['low'].values
    tr2 = np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1))
    tr3 = np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))
    tr1[0] = df_1d['high'].values[0] - df_1d['low'].values[0]  # First bar
    tr2[0] = np.abs(df_1d['high'].values[0] - df_1d['open'].values[0]) if 'open' in df_1d.columns else tr1[0]
    tr3[0] = np.abs(df_1d['low'].values[0] - df_1d['open'].values[0]) if 'open' in df_1d.columns else tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_24 = pd.Series(tr).ewm(span=24, adjust=False, min_periods=24).mean().values
    atr_100 = pd.Series(tr).ewm(span=100, adjust=False, min_periods=100).mean().values
    atr_ratio = atr_24 / (atr_100 + 1e-10)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio, additional_delay_bars=1)
    
    # Williams Alligator components
    # Jaw: EMA13 of median price, smoothed 8 periods
    jaw_raw = ema(median_price, 13)
    jaw = ema(jaw_raw, 8)
    
    # Teeth: EMA8 of median price, smoothed 5 periods
    teeth_raw = ema(median_price, 8)
    teeth = ema(teeth_raw, 5)
    
    # Lips: EMA5 of median price, smoothed 3 periods
    lips_raw = ema(median_price, 5)
    lips = ema(lips_raw, 3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 210  # Need sufficient data for EMA200
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Alligator alignment OR price crosses 1d EMA200 in opposite direction
        if position != 0:
            # Exit long: bearish alignment OR price falls below 1d EMA200
            if position == 1:
                if not (lips[i] > teeth[i] > jaw[i]) or curr_close < ema200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: bullish alignment OR price rises above 1d EMA200
            elif position == -1:
                if not (lips[i] < teeth[i] < jaw[i]) or curr_close > ema200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Alligator alignment with trend filter and volume spike confirmation
        if position == 0:
            # Bullish alignment: Lips > Teeth > Jaw
            bullish_align = lips[i] > teeth[i] and teeth[i] > jaw[i]
            # Bearish alignment: Lips < Teeth < Jaw
            bearish_align = lips[i] < teeth[i] and teeth[i] < jaw[i]
            
            # Volume spike confirmation: ATR(24) > 1.5 * ATR(100)
            vol_spike = atr_ratio_aligned[i] > 1.5
            
            # Long: Bullish alignment AND price > 1d EMA200 AND volume spike
            if bullish_align and curr_close > ema200_1d_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment AND price < 1d EMA200 AND volume spike
            elif bearish_align and curr_close < ema200_1d_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dEMA200_TrendFilter_ATR_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0