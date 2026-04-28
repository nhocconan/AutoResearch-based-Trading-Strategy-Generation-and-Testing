#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d SuperTrend (ATR=10, mult=3.0) for trend direction,
# combined with 6h RSI(14) extremes and volume confirmation (>1.5x 20-bar average).
# Enter long when 1d SuperTrend is bullish (close > upper band) AND 6h RSI < 30 (oversold) AND volume spike.
# Enter short when 1d SuperTrend is bearish (close < lower band) AND 6h RSI > 70 (overbought) AND volume spike.
# Exit when RSI crosses 50 (mean reversion completion) or SuperTrend flips.
# Uses discrete position sizing (0.25) to control risk and minimize fee churn.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.
# SuperTrend effectively captures trend in both bull/bear markets; RSI extremes provide mean-reversion entries within trend.

name = "6h_SuperTrend_RSIExtremes_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for SuperTrend (trend filter)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d ATR(10)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # ATR(10) using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    atr_10_1d = pd.Series(tr).ewm(alpha=1/10, adjust=False, min_periods=10).mean().values
    
    # SuperTrend: Upper Band = (high+low)/2 + mult*ATR, Lower Band = (high+low)/2 - mult*ATR
    hl2 = (high_1d + low_1d) / 2
    mult = 3.0
    upper_band = hl2 + mult * atr_10_1d
    lower_band = hl2 - mult * atr_10_1d
    
    # SuperTrend direction: initialize
    supertrend = np.zeros_like(close_1d)
    direction = np.ones_like(close_1d)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, len(close_1d)):
        if close_1d[i-1] > supertrend[i-1]:
            supertrend[i] = max(upper_band[i], supertrend[i-1])
        else:
            supertrend[i] = min(lower_band[i], supertrend[i-1])
        
        # Update direction
        if close_1d[i] > supertrend[i]:
            direction[i] = 1
        else:
            direction[i] = -1
    
    # Calculate 6h RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Calculate volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    # Align 1d SuperTrend direction to 6h timeframe
    direction_aligned = align_htf_to_ltf(prices, df_1d, direction)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(direction_aligned[i]) or np.isnan(rsi_values[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Trend filter: 1d SuperTrend direction
        bullish_trend = direction_aligned[i] == 1
        bearish_trend = direction_aligned[i] == -1
        
        # RSI extremes
        rsi_val = rsi_values[i]
        rsi_oversold = rsi_val < 30
        rsi_overbought = rsi_val > 70
        rsi_exit_long = rsi_val > 50  # exit long when RSI > 50
        rsi_exit_short = rsi_val < 50  # exit short when RSI < 50
        
        # Entry conditions
        long_entry = bullish_trend and rsi_oversold and vol_confirm
        short_entry = bearish_trend and rsi_overbought and vol_confirm
        
        # Exit conditions
        long_exit = (not bullish_trend) or rsi_exit_long
        short_exit = (not bearish_trend) or rsi_exit_short
        
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