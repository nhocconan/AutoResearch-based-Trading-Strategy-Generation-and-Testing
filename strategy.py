#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Relative Strength Index (RSI) with volume confirmation and volatility filter.
# Long when weekly RSI < 30 (oversold) and price > 1d EMA200 (trend filter) and volume > 1.5x average.
# Short when weekly RSI > 70 (overbought) and price < 1d EMA200 (trend filter) and volume > 1.5x average.
# Exit when RSI returns to neutral zone (40-60) or volatility expands (ATR ratio > 2.0).
# Uses weekly RSI for extreme sentiment reversal, daily EMA200 for trend filter, and volume for confirmation.
# Designed to work in both bull and bear markets by fading extremes only in alignment with higher timeframe trend.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE for RSI
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:  # Need enough for RSI(14)
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate RSI (14) on weekly close
    delta = np.diff(close_1w)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # First average gain/loss
    avg_gain = np.concatenate([np.full(14, np.nan), 
                               [np.mean(gain[:14])] if len(gain) >= 14 else [np.nan]])
    avg_loss = np.concatenate([np.full(14, np.nan), 
                               [np.mean(loss[:14])] if len(loss) >= 14 else [np.nan]])
    
    # Wilder smoothing
    for i in range(15, len(gain)+1):
        if i-1 < len(gain):
            avg_gain = np.append(avg_gain, (avg_gain[-1] * 13 + gain[i-1]) / 14)
            avg_loss = np.append(avg_loss, (avg_loss[-1] * 13 + loss[i-1]) / 14)
    
    # Ensure arrays match length
    if len(avg_gain) < len(close_1w):
        avg_gain = np.concatenate([np.full(len(close_1w) - len(avg_gain), np.nan), avg_gain])
    if len(avg_loss) < len(close_1w):
        avg_loss = np.concatenate([np.full(len(close_1w) - len(avg_loss), np.nan), avg_loss])
    
    # Trim to match close_1w length
    avg_gain = avg_gain[-len(close_1w):]
    avg_loss = avg_loss[-len(close_1w):]
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(avg_loss == 0, 100, rsi)  # Handle no loss case
    rsi = np.where(avg_gain == 0, 0, rsi)    # Handle no gain case
    
    # Calculate 1d EMA200 for trend filter
    close_s = pd.Series(close)
    ema200 = close_s.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate ATR(14) for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    
    # Align indicators to 1d timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi)
    ema200_aligned = align_htf_to_ltf(prices, df_1w, ema200)  # EMA200 is already 1d but aligned for safety
    
    # Volume confirmation: 1.5x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(200, 50)  # Need EMA200 and sufficient lookback
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_aligned[i]) or 
            np.isnan(ema200_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(atr_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Volatility filter: avoid extremely high volatility (ATR > 2x MA)
        vol_filter = atr[i] < 2.0 * atr_ma[i]
        
        if position == 0:
            # Look for RSI extremes in alignment with trend
            # Long: RSI oversold (<30) AND price > EMA200 (uptrend) AND volume confirmation AND vol filter
            if (rsi_aligned[i] < 30 and 
                close[i] > ema200_aligned[i] and 
                volume_confirmed and 
                vol_filter):
                position = 1
                signals[i] = position_size
            # Short: RSI overbought (>70) AND price < EMA200 (downtrend) AND volume confirmation AND vol filter
            elif (rsi_aligned[i] > 70 and 
                  close[i] < ema200_aligned[i] and 
                  volume_confirmed and 
                  vol_filter):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI returns to neutral (40-60) or volatility expands
            if (rsi_aligned[i] >= 40 and rsi_aligned[i] <= 60) or not vol_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI returns to neutral (40-60) or volatility expands
            if (rsi_aligned[i] >= 40 and rsi_aligned[i] <= 60) or not vol_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_WeeklyRSI_EMA200_VolumeVolFilter_v1"
timeframe = "1d"
leverage = 1.0