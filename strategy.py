#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h CRSI + Volume Spike + Choppiness Regime Filter
# Long when CRSI < 15 + volume > 1.5x 20-period average + CHOP > 61.8 (range)
# Short when CRSI > 85 + volume > 1.5x 20-period average + CHOP > 61.8 (range)
# Exit when CRSI crosses back above 50 (long) or below 50 (short)
# Designed for mean reversion in ranging markets with volume confirmation
# Target: 20-50 trades per symbol over 4 years (5-12.5/year)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for CRSI and CHOP calculations
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily RSI(3) for CRSI
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # RSI(3)
    gain_ma = pd.Series(gain).ewm(span=3, adjust=False).mean().values
    loss_ma = pd.Series(loss).ewm(span=3, adjust=False).mean().values
    rs = gain_ma / (loss_ma + 1e-10)
    rsi_3 = 100 - (100 / (1 + rs))
    
    # Calculate RSI streak (2-period)
    up_days = np.where(close_1d[1:] > close_1d[:-1], 1, 0)
    down_days = np.where(close_1d[1:] < close_1d[:-1], 1, 0)
    up_streak = np.where(up_days, np.concatenate([[0], np.cumsum(up_days) * up_days]), 0)
    down_streak = np.where(down_days, np.concatenate([[0], np.cumsum(down_days) * down_days]), 0)
    rsi_streak = 100 * up_streak / (up_streak + down_streak + 1e-10)
    
    # Percentile rank of current RSI(14) over 100 periods
    def rolling_percentile(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i < window - 1:
                result[i] = np.nan
            else:
                window_data = arr[i-window+1:i+1]
                rank = np.sum(window_data <= arr[i]) / window * 100
                result[i] = rank
        return result
    
    # RSI(14) for percentile calculation
    gain_14 = pd.Series(gain).ewm(span=14, adjust=False).mean().values
    loss_14 = pd.Series(loss).ewm(span=14, adjust=False).mean().values
    rs_14 = gain_14 / (loss_14 + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs_14))
    
    percent_rank = rolling_percentile(rsi_14, 100)
    
    # CRSI = (RSI(3) + RSI_streak + PercentRank) / 3
    crsi = (rsi_3 + rsi_streak + percent_rank) / 3
    
    # Calculate Choppiness Index (14-period)
    atr_1 = np.maximum(high[1:] - low[1:], 
                       np.maximum(np.abs(high[1:] - low[:-1]), 
                                  np.abs(low[1:] - high[:-1])))
    atr_1 = np.concatenate([[np.nan], atr_1])
    
    atr_sum = pd.Series(atr_1).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to main timeframe
    crsi_aligned = align_htf_to_ltf(prices, df_1d, crsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 100  # for CRSI and CHOP calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(crsi_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        vol_1d_current = vol_1d[i] if i < len(vol_1d) else vol_1d[-1]
        
        if position == 0:
            # Long setup: CRSI oversold + volume spike + ranging market
            if (crsi_aligned[i] < 15 and 
                vol_1d_current > 1.5 * vol_ma_1d_aligned[i] and  # Volume spike
                chop_aligned[i] > 61.8):                        # Range market
                position = 1
                signals[i] = position_size
            # Short setup: CRSI overbought + volume spike + ranging market
            elif (crsi_aligned[i] > 85 and 
                  vol_1d_current > 1.5 * vol_ma_1d_aligned[i] and  # Volume spike
                  chop_aligned[i] > 61.8):                        # Range market
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: CRSI crosses back above 50
            if crsi_aligned[i] > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: CRSI crosses back below 50
            if crsi_aligned[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_CRSI_Volume_Chop_MeanReversion"
timeframe = "4h"
leverage = 1.0