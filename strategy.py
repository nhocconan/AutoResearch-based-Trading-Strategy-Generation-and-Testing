#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d EMA50 Trend + Volume Spike
# Williams Alligator (JAW/TEETH/LIPS) identifies trend absence when lines intertwine.
# Trend present when LIPS > TEETH > JAW (bull) or LIPS < TEETH < JAW (bear).
# Enter long when Alligator bullish aligned + price > 1d EMA50 + volume spike.
# Enter short when Alligator bearish aligned + price < 1d EMA50 + volume spike.
# Exit when Alligator turns neutral (lines intertwine) or opposite signal.
# Uses discrete position sizing (0.25) to limit drawdown and reduce fee churn.
# Target: 50-150 total trades over 4 years (12-37/year).
# Alligator filters choppy markets, EMA50 provides HTF trend filter, volume confirms conviction.

name = "6h_WilliamsAlligator_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Alligator on 6h chart
    # JAW: 13-period SMMA, shifted 8 bars
    # TEETH: 8-period SMMA, shifted 5 bars
    # LIPS: 5-period SMMA, shifted 3 bars
    # SMMA = smoothed moving average (similar to EMA but with different alpha)
    # We'll use EMA as approximation for SMMA (common in practice)
    
    close_series = pd.Series(close)
    jaw = close_series.ewm(span=13, adjust=False, min_periods=13).mean().shift(8).values
    teeth = close_series.ewm(span=8, adjust=False, min_periods=8).mean().shift(5).values
    lips = close_series.ewm(span=5, adjust=False, min_periods=5).mean().shift(3).values
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13, 8, 5) + 8  # Ensure sufficient history for Alligator and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 1d EMA trend filter
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Alligator conditions
        alligator_bullish = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_bearish = lips[i] < teeth[i] and teeth[i] < jaw[i]
        alligator_neutral = not (alligator_bullish or alligator_bearish)
        
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: Alligator bullish, price > 1d EMA50, volume confirm
            if alligator_bullish and price_above_ema and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: Alligator bearish, price < 1d EMA50, volume confirm
            elif alligator_bearish and price_below_ema and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on Alligator turning neutral or bearish
            if alligator_neutral or alligator_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on Alligator turning neutral or bullish
            if alligator_neutral or alligator_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals