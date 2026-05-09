#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index filter + 1d RSI mean reversion + volume spike
# Uses 1d RSI extremes for mean reversion entries, filtered by 4h choppiness regime
# (CHOP > 61.8 = ranging market) to avoid trending markets where mean reversion fails.
# Volume spike confirms institutional interest at reversal points.
# Designed to work in ranging markets which are common in BTC/ETH consolidation periods.
name = "4h_ChopFilter_1dRSI_MeanRev_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 14-period RSI on daily closes
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (alpha = 1/period)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])  # First average of first 14 periods
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # Align RSI to 4h timeframe (no extra delay needed for RSI as it's based on closed daily bar)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    
    # 4h Choppiness Index (14-period) - measures ranging vs trending markets
    # Higher values indicate ranging markets (good for mean reversion)
    atr_14 = np.zeros(n)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # First TR
    
    for i in range(1, n):
        atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14 if not np.isnan(atr_14[i-1]) else np.mean(tr[max(0, i-13):i+1])
    
    # Handle NaN values in ATR calculation
    for i in range(n):
        if np.isnan(atr_14[i]):
            atr_14[i] = np.mean(tr[max(0, i-13):i+1]) if i >= 1 else tr[0]
    
    # Choppiness Index formula: 100 * log10(sum(ATR14 over period) / (max(high) - min(low))) / log10(period)
    chop_period = 14
    sum_atr = np.zeros(n)
    max_high = np.zeros(n)
    min_low = np.zeros(n)
    
    for i in range(chop_period-1, n):
        sum_atr[i] = np.sum(atr_14[i-chop_period+1:i+1])
        max_high[i] = np.max(high[i-chop_period+1:i+1])
        min_low[i] = np.min(low[i-chop_period+1:i+1])
    
    # Avoid division by zero
    range_hl = max_high - min_low
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    
    chop = 100 * np.log10(sum_atr / range_hl) / np.log10(chop_period)
    chop = np.where(np.isnan(chop), 50, chop)  # Default to middle range if calculation fails
    
    # Volume spike filter: volume > 2.0x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, chop_period-1)  # Ensure we have enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(chop[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: RSI oversold (<30) in ranging market (CHOP > 61.8) with volume spike
            if (rsi_1d_aligned[i] < 30 and chop[i] > 61.8 and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (>70) in ranging market (CHOP > 61.8) with volume spike
            elif (rsi_1d_aligned[i] > 70 and chop[i] > 61.8 and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI returns to neutral (>45) or stoploss via opposite signal
            if rsi_1d_aligned[i] > 45:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI returns to neutral (<55) or stoploss via opposite signal
            if rsi_1d_aligned[i] < 55:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals