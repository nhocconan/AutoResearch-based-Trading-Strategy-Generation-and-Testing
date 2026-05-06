#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Donchian channel breakout with ATR volatility filter
# Long when price breaks above 1w Donchian upper (20-period) AND ATR(14) < ATR(50) (low volatility environment)
# Short when price breaks below 1w Donchian lower (20-period) AND ATR(14) < ATR(50)
# Exit when price crosses 1w EMA20 (mean reversion to weekly trend)
# Uses discrete sizing 0.25 to balance return and drawdown
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# 1w Donchian provides clear structure with proven breakout edge in BTC/ETH
# ATR filter ensures we only trade during low volatility periods (reduces false breakouts)
# Works in both bull and bear markets by capturing breakouts with volatility filter

name = "1d_1wDonchian20_ATRFilter_EMA20Exit_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data ONCE before loop for Donchian channels and EMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need sufficient data for Donchian(20)
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Donchian channel (20-period)
    high_series_1w = pd.Series(high_1w)
    low_series_1w = pd.Series(low_1w)
    donchian_upper_1w = high_series_1w.rolling(window=20, min_periods=20).max().values
    donchian_lower_1w = low_series_1w.rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w EMA20 for exit signal
    close_series_1w = pd.Series(close_1w)
    ema_20_1w = close_series_1w.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 1w Donchian and EMA to 1d timeframe (wait for completed 1w bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper_1w)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower_1w)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate ATR(14) and ATR(50) for volatility filter on 1d timeframe
    # True Range calculation
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0  # First value has no previous close
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR calculation using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_50 = pd.Series(tr).ewm(alpha=1/50, adjust=False, min_periods=50).mean().values
    
    # Volatility filter: ATR(14) < ATR(50) (low volatility environment)
    vol_filter = atr_14 < atr_50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(atr_14[i]) or np.isnan(atr_50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1w Donchian upper with low volatility filter
            if (close[i] > donchian_upper_aligned[i] and close[i-1] <= donchian_upper_aligned[i-1] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1w Donchian lower with low volatility filter
            elif (close[i] < donchian_lower_aligned[i] and close[i-1] >= donchian_lower_aligned[i-1] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1w EMA20 (mean reversion to weekly trend)
            if close[i] < ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1w EMA20 (mean reversion to weekly trend)
            if close[i] > ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals