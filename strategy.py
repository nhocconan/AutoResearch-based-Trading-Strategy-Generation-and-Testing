#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day RSI mean reversion with weekly ADX trend filter and volume confirmation
# Long when RSI(14) < 30 AND weekly ADX > 25 AND volume > 1.5x 20-day average
# Short when RSI(14) > 70 AND weekly ADX > 25 AND volume > 1.5x 20-day average
# Exit when RSI crosses back above 50 (long) or below 50 (short)
# Uses RSI for mean reversion in ranging markets, ADX to filter strong trends, volume for confirmation
# Target: 30-100 total trades over 4 years (7-25/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Calculate weekly ADX(14) for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.concatenate([[close_1w[0]], close_1w[:-1]]))
    tr3 = np.abs(low_1w - np.concatenate([[close_1w[0]], close_1w[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.concatenate([[high_1w[0]], high_1w[:-1]])) > 
                       (np.concatenate([[low_1w[0]], low_1w[:-1]]) - low_1w), 
                       np.maximum(high_1w - np.concatenate([[high_1w[0]], high_1w[:-1]]), 0), 0)
    dm_minus = np.where((np.concatenate([[low_1w[0]], low_1w[:-1]]) - low_1w) > 
                        (high_1w - np.concatenate([[high_1w[0]], high_1w[:-1]])), 
                        np.maximum(np.concatenate([[low_1w[0]], low_1w[:-1]]) - low_1w, 0), 0)
    
    # Smooth with Wilder's smoothing (alpha=1/14)
    tr14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / (tr14 + 1e-10)
    di_minus = 100 * dm_minus_14 / (tr14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    adx_values = adx.values
    
    # Align weekly ADX to daily
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx_values)
    
    # Calculate volume average for confirmation (20-day)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 30
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi[i]) or np.isnan(adx_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        rsi_val = rsi[i]
        adx_val = adx_aligned[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: RSI oversold + strong trend + volume confirmation
            if (rsi_val < 30 and adx_val > 25 and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: RSI overbought + strong trend + volume confirmation
            elif (rsi_val > 70 and adx_val > 25 and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI crosses back above 50
            if rsi_val > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI crosses back below 50
            if rsi_val < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_RSI_ADX_Volume_MeanReversion"
timeframe = "1d"
leverage = 1.0