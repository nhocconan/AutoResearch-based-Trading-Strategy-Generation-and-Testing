#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean reversion with 4h/1d regime filter
# Long when: price < 4h Bollinger lower band (20,2) AND 1d close > 1d EMA200 (bullish regime) AND volume > 1.5x 20-period avg
# Short when: price > 4h Bollinger upper band (20,2) AND 1d close < 1d EMA200 (bearish regime) AND volume > 1.5x 20-period avg
# Exit when price crosses 4h Bollinger middle band (20-period SMA)
# Uses 4h for Bollinger bands (structure) and 1d EMA200 for regime, 1h only for entry timing
# Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag
# Session filter: 08-20 UTC to avoid low-liquidity hours

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h and daily data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate 4h Bollinger Bands (20,2)
    close_4h = df_4h['close'].values
    sma_20 = pd.Series(close_4h).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_4h).rolling(window=20, min_periods=20).std().values
    bband_upper = sma_20 + 2 * std_20
    bband_middle = sma_20
    bband_lower = sma_20 - 2 * std_20
    
    # Calculate 1d EMA200 for regime filter
    close_daily = df_daily['close'].values
    ema_200 = pd.Series(close_daily).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 4h volume average (20-period)
    vol_4h = df_4h['volume'].values
    vol_ma_20 = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 1h timeframe
    bband_upper_aligned = align_htf_to_ltf(prices, df_4h, bband_upper)
    bband_middle_aligned = align_htf_to_ltf(prices, df_4h, bband_middle)
    bband_lower_aligned = align_htf_to_ltf(prices, df_4h, bband_lower)
    ema_200_aligned = align_htf_to_ltf(prices, df_daily, ema_200)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for calculations (200 for EMA200)
    start = 200
    
    for i in range(start, n):
        # Skip if any critical data is NaN or outside session
        if (np.isnan(bband_upper_aligned[i]) or np.isnan(bband_lower_aligned[i]) or 
            np.isnan(ema_200_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_1h = volume[i]  # Current 1h volume
        
        if position == 0:
            # Long setup: price < 4h Bollinger lower band AND bullish regime (price > EMA200) AND volume confirmation
            if (price < bband_lower_aligned[i] and 
                price > ema_200_aligned[i] and  # Bullish regime: above daily EMA200
                vol_1h > 1.5 * vol_ma_20_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short setup: price > 4h Bollinger upper band AND bearish regime (price < EMA200) AND volume confirmation
            elif (price > bband_upper_aligned[i] and 
                  price < ema_200_aligned[i] and  # Bearish regime: below daily EMA200
                  vol_1h > 1.5 * vol_ma_20_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses above 4h Bollinger middle band
            if price > bband_middle_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses below 4h Bollinger middle band
            if price < bband_middle_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_Bollinger_MeanReversion_Regime"
timeframe = "1h"
leverage = 1.0