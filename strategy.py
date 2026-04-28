#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Williams Alligator (Jaw/Teeth/Lips) with EMA trend filter and volume confirmation.
# Enter long when price > Alligator Lips and EMA50 > EMA200 (bullish alignment) with volume > 1.5x 20-bar average.
# Enter short when price < Alligator Lips and EMA50 < EMA200 (bearish alignment) with volume > 1.5x 20-bar average.
# Uses discrete position sizing (0.25) to limit drawdown. Target: 10-25 trades/year.
# Alligator identifies trend presence and direction, EMA alignment filters false signals, volume confirms strength.
# Works in bull (trend continuation) and bear (trend reversal) markets by following the Alligator's alignment.

name = "1d_WilliamsAlligator_EMA_Trend_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Williams Alligator (HTF)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w Williams Alligator: SMAs of median price
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    # SMMA (Smoothed Moving Average) = EMA with alpha = 1/period
    median_price = (df_1w['high'] + df_1w['low']) / 2.0
    
    jaw = pd.Series(median_price).ewm(alpha=1/13, adjust=False).mean().values
    teeth = pd.Series(median_price).ewm(alpha=1/8, adjust=False).mean().values
    lips = pd.Series(median_price).ewm(alpha=1/5, adjust=False).mean().values
    
    # Align 1w Alligator to 1d timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # Calculate 1d EMAs for trend filter
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = close_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 1d volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure sufficient history for EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_50[i]) or np.isnan(ema_200[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Bullish alignment: price > Lips and EMA50 > EMA200
        bullish_alignment = close[i] > lips_aligned[i] and ema_50[i] > ema_200[i]
        # Bearish alignment: price < Lips and EMA50 < EMA200
        bearish_alignment = close[i] < lips_aligned[i] and ema_50[i] < ema_200[i]
        
        # Entry conditions with volume confirmation
        long_entry = bullish_alignment and volume_confirm[i]
        short_entry = bearish_alignment and volume_confirm[i]
        
        # Exit conditions: opposite alignment
        long_exit = not bullish_alignment  # Price <= Lips or EMA50 <= EMA200
        short_exit = not bearish_alignment  # Price >= Lips or EMA50 >= EMA200
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals