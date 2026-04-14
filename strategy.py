#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for trend and ATR
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate True Range and ATR(14) on weekly
    tr = np.zeros(len(df_1w))
    tr[0] = high_1w[0] - low_1w[0]
    for i in range(1, len(df_1w)):
        tr[i] = max(
            high_1w[i] - low_1w[i],
            abs(high_1w[i] - close_1w[i-1]),
            abs(low_1w[i] - close_1w[i-1])
        )
    
    atr_1w = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 14:
        atr_1w[13] = np.mean(tr[:14])
        for i in range(14, len(df_1w)):
            atr_1w[i] = (atr_1w[i-1] * 13 + tr[i]) / 14
    
    # Align 1w ATR to daily timeframe
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Weekly EMA(20) for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_20_1w = close_1w_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Daily Donchian(20) breakout
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike detection (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_1w_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or
            np.isnan(high_20[i]) or
            np.isnan(low_20[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.3% of price)
        if atr_1w_aligned[i] < 0.003 * close[i]:
            signals[i] = 0.0
            continue
        
        # Volume ratio: current daily volume vs 20-period average
        if vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        # Volume threshold: require significant spike
        vol_threshold = 2.0
        
        if position == 0:
            # Long: Price breaks above Donchian high with volume spike and weekly uptrend
            if (close[i] > high_20[i] and 
                volume_ratio > vol_threshold and
                close[i] > ema_20_1w_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below Donchian low with volume spike and weekly downtrend
            elif (close[i] < low_20[i] and 
                  volume_ratio > vol_threshold and
                  close[i] < ema_20_1w_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price closes below Donchian low (trend reversal)
            if close[i] < low_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price closes above Donchian high (trend reversal)
            if close[i] > high_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_WeeklyTrend_Donchian_Volume"
timeframe = "1d"
leverage = 1.0