#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian breakout with 4h trend filter and 1d volatility filter
# Long when price breaks above 20-period high AND 4h close above 4h EMA(20) AND 1d ATR ratio > 0.8
# Short when price breaks below 20-period low AND 4h close below 4h EMA(20) AND 1d ATR ratio > 0.8
# Uses 4h trend filter to avoid counter-trend trades, 1d ATR filter to avoid low volatility periods
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# Uses discrete sizing (0.20) to minimize churn and manage drawdown

name = "1h_donchian20_4h_ema_1d_vol_filter_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian Channel (20-period) on 1h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    donchian_high = high_series.rolling(window=20, min_periods=20).max()
    donchian_low = low_series.rolling(window=20, min_periods=20).min()
    
    # 4h trend filter: EMA(20) on 4h close
    df_4h = get_htf_data(prices, '4h')
    four_hour_close = df_4h['close'].values
    
    # Calculate 20-period EMA on 4h close
    four_hour_close_series = pd.Series(four_hour_close)
    four_hour_ema = four_hour_close_series.ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Align 4h EMA to 1h timeframe
    four_hour_ema_aligned = align_htf_to_ltf(prices, df_4h, four_hour_ema)
    
    # 1d volatility filter: ATR ratio (current ATR / 20-period ATR average)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR(14) on 1d
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_1d = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_ratio_1d = atr_1d / atr_ma_1d
    
    # Align 1d ATR ratio to 1h timeframe
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(four_hour_ema_aligned[i]) or 
            np.isnan(atr_ratio_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        if hours[i] < 8 or hours[i] > 20:
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: reverse position or trend change or volatility drop
        if position == 1:  # long position
            # Exit: price breaks below 20-period low or 4h trend turns bearish or low volatility
            if (close[i] <= donchian_low[i] or 
                four_hour_close[i] < four_hour_ema_aligned[i] or
                atr_ratio_1d_aligned[i] < 0.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: price breaks above 20-period high or 4h trend turns bullish or low volatility
            if (close[i] >= donchian_high[i] or 
                four_hour_close[i] > four_hour_ema_aligned[i] or
                atr_ratio_1d_aligned[i] < 0.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with 4h trend filter and volatility filter
            # Long: price breaks above 20-period high AND 4h close above 4h EMA AND sufficient volatility
            if (close[i] > donchian_high[i] and 
                four_hour_close[i] > four_hour_ema_aligned[i] and
                atr_ratio_1d_aligned[i] >= 0.8):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 20-period low AND 4h close below 4h EMA AND sufficient volatility
            elif (close[i] < donchian_low[i] and 
                  four_hour_close[i] < four_hour_ema_aligned[i] and
                  atr_ratio_1d_aligned[i] >= 0.8):
                signals[i] = -0.20
                position = -1
    
    return signals