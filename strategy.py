#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h trend filter, volume confirmation, and ATR stoploss
# Donchian channels provide robust price breakout signals; 12h EMA21 filters for higher timeframe trend alignment
# Volume > 1.5x 20-period average confirms institutional participation
# ATR-based stoploss limits downside risk
# Designed for 4h timeframe with selective entries to avoid overtrading
# Target: 19-50 trades per year per symbol (75-200 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 21-period EMA on 12h timeframe for trend filter
    ema21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema21_12h)
    
    # Calculate Donchian channels (20-period high/low)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian upper/lower bands
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR(14) for volatility and stoploss
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    true_range[0] = high_low[0]  # First value
    atr = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(ema21_12h_aligned[i]) or \
           np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(atr[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 12h trend
        is_uptrend = close[i] > ema21_12h_aligned[i]
        is_downtrend = close[i] < ema21_12h_aligned[i]
        
        # Volume confirmation
        has_volume = vol_filter[i]
        
        price = close[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian high + uptrend + volume
            long_signal = (price > donchian_high[i]) and is_uptrend and has_volume
            
            # Short entry: price breaks below Donchian low + downtrend + volume
            short_signal = (price < donchian_low[i]) and is_downtrend and has_volume
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price breaks below Donchian low OR 2*ATR stoploss
            if price < donchian_low[i] or price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high OR 2*ATR stoploss
            if price > donchian_high[i] or price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hTrendFilter_Volume_ATR"
timeframe = "4h"
leverage = 1.0