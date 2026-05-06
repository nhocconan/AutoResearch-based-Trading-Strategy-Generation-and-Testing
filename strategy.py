#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation
# Long when Alligator jaws (13) < teeth (8) < lips (5) AND close > 1d EMA50 AND volume > 1.5 * 20-bar avg volume
# Short when Alligator jaws > teeth > lips AND close < 1d EMA50 AND volume > 1.5 * 20-bar avg volume
# Exit when Alligator lines re-interlace (jaws crosses teeth or lips) indicating loss of trend momentum
# Uses discrete sizing 0.25 to control fee drag and drawdown
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Williams Alligator identifies strong trending markets via aligned SMAs; EMA50 ensures higher-timeframe trend alignment
# Volume spike confirms institutional participation; exit on Alligator re-interlace works in both trending and ranging markets
# Specifically designed for 12h timeframe to reduce trade frequency vs lower timeframes while capturing significant moves

name = "12h_WilliamsAlligator_1dEMA50_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator for 12h timeframe (SMAs of median price)
    # Median price = (high + low) / 2
    median_price = (high + low) / 2.0
    
    # Jaw: 13-period SMMA (smoothed moving average) of median price
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    # Teeth: 8-period SMMA of median price
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    # Lips: 5-period SMMA of median price
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 12h timeframe (wait for completed HTF bar)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate volume confirmation: volume > 1.5 * 20-bar average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Alligator signals with trend and volume filters
            # Long: Jaw < Teeth < Lips (aligned for uptrend) AND uptrend AND volume spike
            if jaw[i] < teeth[i] and teeth[i] < lips[i] and close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Jaw > Teeth > Lips (aligned for downtrend) AND downtrend AND volume spike
            elif jaw[i] > teeth[i] and teeth[i] > lips[i] and close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator lines re-interlace (Jaw crosses Teeth or Lips) indicating trend weakness
            if jaw[i] >= teeth[i] or jaw[i] >= lips[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator lines re-interlace (Jaw crosses Teeth or Lips) indicating trend weakness
            if jaw[i] <= teeth[i] or jaw[i] <= lips[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals