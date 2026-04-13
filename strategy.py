#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d volume confirmation and 1w trend filter.
# Williams Alligator identifies trend presence and direction via three smoothed moving averages (Jaws, Teeth, Lips).
# The strategy enters when the Alligator "wakes up" (lines diverge) in the direction of the weekly trend.
# Volume confirmation on the 1d timeframe ensures breakouts have institutional participation.
# Designed to work in both bull (trend following) and bear (counter-trend on weekly reversals) markets.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Williams Alligator on 12h price data (Jaws=13, Teeth=8, Lips=5)
    def smma(data, period):
        """Smoothed Moving Average"""
        sma = np.full(len(data), np.nan)
        if len(data) >= period:
            sma[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                sma[i] = (sma[i-1] * (period-1) + data[i]) / period
        return sma
    
    jaws = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Alligator signals: 
    # - Bullish: Lips > Teeth > Jaws (green alignment, mouth opening up)
    # - Bearish: Lips < Teeth < Jaws (red alignment, mouth opening down)
    # - Sleeping: intertwined (no trend)
    bullish_alignment = (lips > teeth) & (teeth > jaws)
    bearish_alignment = (lips < teeth) & (teeth < jaws)
    
    # Volume confirmation: 1d volume > 1.5x 20-period average
    vol_1d = df_1d['volume'].values
    vol_ma_20 = smma(vol_1d, 20) if len(vol_1d) >= 20 else np.full(len(vol_1d), np.nan)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    volume_confirmed = volume > (vol_ma_20_aligned * 1.5)
    
    # Weekly trend filter: price relative to weekly Alligator (using weekly close)
    close_1w = df_1w['close'].values
    jaws_1w = smma(close_1w, 13)
    teeth_1w = smma(close_1w, 8)
    lips_1w = smma(close_1w, 5)
    # Weekly bullish: price above all three lines
    weekly_bullish = close_1w > jaws_1w
    # Weekly bearish: price below all three lines
    weekly_bearish = close_1w < jaws_1w
    jaws_1w_aligned = align_htf_to_ltf(prices, df_1w, jaws_1w)
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish)
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish)
    
    # Align Alligator lines to 12h timeframe
    jaws_aligned = align_htf_to_ltf(prices, None, jaws)  # Same timeframe, no alignment needed
    teeth_aligned = align_htf_to_ltf(prices, None, teeth)
    lips_aligned = align_htf_to_ltf(prices, None, lips)
    bullish_aligned = align_htf_to_ltf(prices, None, bullish_alignment)
    bearish_aligned = align_htf_to_ltf(prices, None, bearish_alignment)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(volume_confirmed[i]) or
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        volume_ok = volume_confirmed[i]
        weekly_up = weekly_bullish_aligned[i]
        weekly_down = weekly_bearish_aligned[i]
        bullish = bullish_aligned[i]
        bearish = bearish_aligned[i]
        
        if position == 0:
            # Long: Alligator bullish alignment, volume confirmed, weekly bullish trend
            if bullish and volume_ok and weekly_up:
                position = 1
                signals[i] = position_size
            # Short: Alligator bearish alignment, volume confirmed, weekly bearish trend
            elif bearish and volume_ok and weekly_down:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Alligator turns bearish OR volume dries up OR weekly trend turns bearish
            if (not bullish or not volume_ok or not weekly_up):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Alligator turns bullish OR volume dries up OR weekly trend turns bullish
            if (not bearish or not volume_ok or not weekly_down):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_1w_Williams_Alligator_Volume_Trend"
timeframe = "12h"
leverage = 1.0