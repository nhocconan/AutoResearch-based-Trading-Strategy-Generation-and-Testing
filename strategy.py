#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h/1d for directional bias and 1h for precise entry timing
# Uses 4h Supertrend (ATR=10, mult=3) for primary trend direction
# Uses 1d RSI(14) with extreme thresholds (RSI<30 for long, RSI>70 for short) for momentum extremes
# Enters only during 08-20 UTC session with volume confirmation (volume > 1.2x 20-period MA)
# Target: 15-30 trades per year (60-120 over 4 years) with 0.20 position sizing
# Designed to work in both bull and bear markets by combining trend following with mean reversion extremes

name = "1h_4hSupertrend_1dRSI_Extreme_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h Supertrend (ATR=10, mult=3)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 11:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # ATR(10)
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2 = (high_4h + low_4h) / 2
    upper_band = hl2 + (3 * atr_10)
    lower_band = hl2 - (3 * atr_10)
    
    upper_band_final = np.full_like(close_4h, np.nan)
    lower_band_final = np.full_like(close_4h, np.nan)
    supertrend = np.full_like(close_4h, np.nan)
    direction = np.full_like(close_4h, 1)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_4h)):
        if np.isnan(atr_10[i]) or np.isnan(atr_10[i-1]):
            upper_band_final[i] = upper_band[i]
            lower_band_final[i] = lower_band[i]
        else:
            if close_4h[i-1] > upper_band_final[i-1]:
                upper_band_final[i] = upper_band[i]
            else:
                upper_band_final[i] = min(upper_band[i], upper_band_final[i-1])
            
            if close_4h[i-1] < lower_band_final[i-1]:
                lower_band_final[i] = lower_band[i]
            else:
                lower_band_final[i] = max(lower_band[i], lower_band_final[i-1])
        
        if np.isnan(upper_band_final[i]) or np.isnan(lower_band_final[i]):
            supertrend[i] = np.nan
            direction[i] = direction[i-1] if i > 0 else 1
        else:
            if close_4h[i] <= upper_band_final[i]:
                supertrend[i] = upper_band_final[i]
                direction[i] = -1
            else:
                supertrend[i] = lower_band_final[i]
                direction[i] = 1
    
    # Align Supertrend direction to 1h timeframe
    supertrend_direction = align_htf_to_ltf(prices, df_4h, direction)
    
    # Calculate 1d RSI(14)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 15:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d)
    delta = np.concatenate([[np.nan], delta])
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # Align RSI to 1h timeframe
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    
    # Volume confirmation: >1.2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.2 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(supertrend_direction[i]) or np.isnan(rsi_14_aligned[i]) or 
            np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 4h uptrend AND 1d RSI < 30 (oversold) with volume confirmation
            if supertrend_direction[i] == 1 and rsi_14_aligned[i] < 30 and volume_filter[i]:
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend AND 1d RSI > 70 (overbought) with volume confirmation
            elif supertrend_direction[i] == -1 and rsi_14_aligned[i] > 70 and volume_filter[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: 4h downtrend OR 1d RSI > 70 (overbought)
            if supertrend_direction[i] == -1 or rsi_14_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: 4h uptrend OR 1d RSI < 30 (oversold)
            if supertrend_direction[i] == 1 or rsi_14_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals