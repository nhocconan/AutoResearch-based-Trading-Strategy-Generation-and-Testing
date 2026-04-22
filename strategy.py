#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 12h EMA trend filter and volume confirmation
# Long when Jaw < Teeth < Lips (bullish alignment) + close > 12h EMA50 + volume spike
# Short when Jaw > Teeth > Lips (bearish alignment) + close < 12h EMA50 + volume spike
# Exit when Alligator alignment breaks or trend reverses
# Williams Alligator uses smoothed moving averages (SMMA) to identify trends
# Designed for low trade frequency (~20-40/year) to minimize fee drain.
# Works in bull/bear by combining trend-following with Alligator alignment and volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 50-period EMA on 12h close for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Williams Alligator on 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Median price = (High + Low) / 2
    median_price = (high + low) / 2
    
    # Smoothed Moving Average (SMMA) function
    def smma(data, period):
        sma = np.full_like(data, np.nan, dtype=float)
        sma[period-1] = np.nanmean(data[:period])
        for i in range(period, len(data)):
            sma[i] = (sma[i-1] * (period-1) + data[i]) / period
        return sma
    
    # Alligator lines: Jaw (13-period SMMA, 8-bar shift), Teeth (8-period SMMA, 5-bar shift), Lips (5-period SMMA, 3-bar shift)
    jaw = smma(median_price, 13)
    jaw = np.roll(jaw, 8)  # Shift forward by 8 bars
    teeth = smma(median_price, 8)
    teeth = np.roll(teeth, 5)  # Shift forward by 5 bars
    lips = smma(median_price, 5)
    lips = np.roll(lips, 3)  # Shift forward by 3 bars
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        vol_spike = vol > 2.0 * vol_ma
        
        # Alligator alignment
        bullish_alignment = jaw[i] < teeth[i] < lips[i]
        bearish_alignment = jaw[i] > teeth[i] > lips[i]
        
        if position == 0:
            # Long conditions: bullish alignment + uptrend + volume spike
            if bullish_alignment and price > ema_50_12h_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: bearish alignment + downtrend + volume spike
            elif bearish_alignment and price < ema_50_12h_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: Alligator alignment breaks or trend reverses
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when bullish alignment breaks or trend turns down
                if not bullish_alignment or price < ema_50_12h_aligned[i]:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when bearish alignment breaks or trend turns up
                if not bearish_alignment or price > ema_50_12h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsAlligator_12hEMA50_Volume"
timeframe = "6h"
leverage = 1.0