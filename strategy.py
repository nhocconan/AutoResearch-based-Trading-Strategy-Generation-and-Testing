#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation.
# Long when Alligator jaws (13) < teeth (8) < lips (5) AND 1d close > 1d EMA34 (uptrend) AND 12h volume > 1.5x 20-period volume MA.
# Short when Alligator jaws (13) > teeth (8) > lips (5) AND 1d close < 1d EMA34 (downtrend) AND 12h volume > 1.5x 20-period volume MA.
# Exit on Alligator convergence (jaws-teeth < 0.5% of price) or trend reversal.
# Uses 12h timeframe to achieve 50-150 total trades over 4 years (12-37/year) with strict entry conditions.
# Williams Alligator identifies trending vs ranging markets, 1d EMA34 filters for higher-timeframe trend alignment, volume confirms participation.
# Works in both bull and bear markets by only trading in the direction of the 1d trend when Alligator is aligned and volume confirms.

name = "12h_WilliamsAlligator_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend direction
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Alligator on 12h timeframe
    # Jaws: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    def smma(source, length):
        # Smoothed Moving Average (SMMA) - similar to EMA but with different smoothing
        if length < 1:
            return np.full_like(source, np.nan)
        result = np.full_like(source, np.nan)
        alpha = 1.0 / length
        for i in range(len(source)):
            if np.isnan(source[i]):
                result[i] = np.nan
            elif i == 0:
                result[i] = source[i]
            else:
                if np.isnan(result[i-1]):
                    result[i] = source[i]
                else:
                    result[i] = result[i-1] + alpha * (source[i] - result[i-1])
        return result
    
    jaws = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Calculate 12h volume 20-period MA for spike detection
    volume_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaws[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(volume_ma_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        
        # Volume spike condition: current 12h volume > 1.5x 20-period volume MA
        volume_spike = volume[i] > (volume_ma_12h[i] * 1.5)
        
        # Alligator alignment conditions
        alligator_long = jaws[i] < teeth[i] and teeth[i] < lips[i]  # Jaws < Teeth < Lips (uptrend alignment)
        alligator_short = jaws[i] > teeth[i] and teeth[i] > lips[i]  # Jaws > Teeth > Lips (downtrend alignment)
        
        # 1d trend conditions
        trend_up = close_val > ema_34_1d_aligned[i]   # 1d uptrend
        trend_down = close_val < ema_34_1d_aligned[i]  # 1d downtrend
        
        # Alligator convergence (exit condition): jaws-teeth < 0.5% of price
        alligator_convergence = np.abs(jaws[i] - teeth[i]) < (close_val * 0.005)
        
        if position == 0:
            # Long: Alligator long alignment AND 1d uptrend AND volume spike
            if alligator_long and trend_up and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Alligator short alignment AND 1d downtrend AND volume spike
            elif alligator_short and trend_down and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator convergence OR trend changes
            if alligator_convergence or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator convergence OR trend changes
            if alligator_convergence or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals