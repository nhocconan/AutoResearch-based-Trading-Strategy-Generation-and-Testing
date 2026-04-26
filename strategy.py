#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_RegimeFilter
Hypothesis: On 4h timeframe, enter long when price breaks above Camarilla R3 level AND 1d trend is up (close > EMA34) AND volume > 1.8x 20-period average AND choppiness regime is trending (CHOP < 40). Enter short when price breaks below S3 level AND 1d trend is down (close < EMA34) AND volume spike AND CHOP < 40. Uses Camarilla pivot levels for precise support/resistance, 1d EMA34 for higher timeframe trend alignment, volume confirmation for institutional participation, and choppiness filter to avoid whipsaws in ranging markets. Designed for moderate trade frequency (20-40/year) to balance opportunity and fee drag while capturing strong trends in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Daily Camarilla Pivot Levels (R3, S3)
    # Based on previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R4 = close + ((high-low)*1.1/2), R3 = close + ((high-low)*1.1/4)
    # S3 = close - ((high-low)*1.1/4), S4 = close - ((high-low)*1.1/2)
    camarilla_r3 = close_1d + ((high_1d - low_1d) * 1.1 / 4)
    camarilla_s3 = close_1d - ((high_1d - low_1d) * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: volume > 1.8x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 1.8
    
    # Choppiness Index regime filter (trending when CHOP < 40)
    # CHOP = 100 * log10(sum(ATR(14)) / (max(high, n) - min(low, n))) / log10(n)
    # Simplified: CHOP < 40 = trending, CHOP > 61.8 = ranging
    atr_period = 14
    chop_period = 14
    tr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]  # First TR
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Sum of ATR over chop_period
    atr_sum = pd.Series(atr).rolling(window=chop_period, min_periods=chop_period).sum().values
    
    # Max high and min low over chop_period
    max_high = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    min_low = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    
    # Avoid division by zero
    range_hl = max_high - min_low
    range_hl = np.maximum(range_hl, 1e-10)
    
    # Chop calculation
    chop = 100 * np.log10(atr_sum / range_hl) / np.log10(chop_period)
    chop_trending = chop < 40.0  # Trending regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA warmup (34), ATR warmup (14+14=28), volume MA warmup (20)
    start_idx = max(34, 28, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Breakout conditions relative to Camarilla levels
        breakout_above_r3 = close[i] > camarilla_r3_aligned[i]
        breakout_below_s3 = close[i] < camarilla_s3_aligned[i]
        
        # 1d trend filter
        trend_uptrend = close[i] > ema_34_1d_aligned[i]
        trend_downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price above R3 + 1d uptrend + volume spike + trending regime
            long_signal = breakout_above_r3 and trend_uptrend and volume_spike[i] and chop_trending[i]
            
            # Short: price below S3 + 1d downtrend + volume spike + trending regime
            short_signal = breakout_below_s3 and trend_downtrend and volume_spike[i] and chop_trending[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price breaks below S3 OR trend change to downtrend
            if close[i] < camarilla_s3_aligned[i] or not trend_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above R3 OR trend change to uptrend
            if close[i] > camarilla_r3_aligned[i] or not trend_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_RegimeFilter"
timeframe = "4h"
leverage = 1.0