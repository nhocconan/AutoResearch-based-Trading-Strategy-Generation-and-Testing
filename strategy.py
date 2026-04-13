#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout + 1d HMA trend filter + volume confirmation
    # Enter long when price breaks above Donchian(20) high AND 1d HMA(21) rising AND volume > 1.5x 20-bar avg
    # Enter short when price breaks below Donchian(20) low AND 1d HMA(21) falling AND volume > 1.5x 20-bar avg
    # Exit on opposite Donchian(10) break or ATR-based stoploss
    # Uses 1d HTF for HMA trend (more stable than 4h) and 4h for entry/exit timing
    # Donchian provides clear structure, HMA filters trend direction, volume confirms participation
    # Works in bull (continuation breaks with trend) and bear (reversal breaks against trend)
    # Target: 75-200 total trades over 4 years (19-50/year) to balance edge and fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for primary timeframe
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Get 1d data for HMA trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d HMA(21) - Hull Moving Average
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    hlen = len(close_1d)
    hma_1d = np.full(hlen, np.nan)
    if hlen >= 21:
        half_n = 21 // 2
        sqrt_n = int(np.sqrt(21))
        wma_half = np.array([wma(close_1d[i:i+half_n], half_n) if i+half_n <= hlen else np.nan 
                            for i in range(hlen)])
        wma_full = np.array([wma(close_1d[i:i+21], 21) if i+21 <= hlen else np.nan 
                            for i in range(hlen)])
        raw_hma = 2 * wma_half - wma_full
        hma_1d = np.array([wma(raw_hma[i:i+sqrt_n], sqrt_n) if i+sqrt_n <= len(raw_hma) else np.nan 
                          for i in range(len(raw_hma))])
        # Align lengths
        hma_1d_full = np.full(hlen, np.nan)
        start_idx = half_n + sqrt_n - 1
        if start_idx < hlen:
            end_idx = min(start_idx + len(hma_1d), hlen)
            hma_1d_full[start_idx:end_idx] = hma_1d[:end_idx-start_idx]
        hma_1d = hma_1d_full
    
    # Align 1d HMA to 4h timeframe
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate Donchian channels on 4h
    def donchian_channel(high, low, window):
        upper = pd.Series(high).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    donchian_20_upper, donchian_20_lower = donchian_channel(high_4h, low_4h, 20)
    donchian_10_upper, donchian_10_lower = donchian_channel(high_4h, low_4h, 10)
    
    # Align Donchian levels to LTF (15m) prices
    donchian_20_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_20_upper)
    donchian_20_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_20_lower)
    donchian_10_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_10_upper)
    donchian_10_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_10_lower)
    
    # Volume confirmation: volume > 1.5x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * avg_volume)
    
    # ATR for stoploss (using 4h ATR(14))
    def atr(high, low, close, window):
        tr1 = pd.Series(high).rolling(window=2).max().values - pd.Series(low).rolling(window=2).min().values
        tr2 = np.abs(pd.Series(high).rolling(window=2).shift(1).values - pd.Series(close).rolling(window=2).shift(1).values)
        tr3 = np.abs(pd.Series(low).rolling(window=2).shift(1).values - pd.Series(close).rolling(window=2).shift(1).values)
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        atr_vals = pd.Series(tr).rolling(window=window, min_periods=window).mean().values
        return atr_vals
    
    atr_14 = atr(high_4h, low_4h, close_4h, 14)
    atr_14_aligned = align_htf_to_ltf(prices, df_4h, atr_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):  # start from 20 to ensure Donchian(20) is ready
        # Skip if data not ready
        if (np.isnan(hma_1d_aligned[i]) or np.isnan(donchian_20_upper_aligned[i]) or 
            np.isnan(donchian_20_lower_aligned[i]) or np.isnan(atr_14_aligned[i]) or
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_20_upper_aligned[i-1]  # break above previous Donchian(20) high
        breakout_down = close[i] < donchian_20_lower_aligned[i-1]  # break below previous Donchian(20) low
        
        # HMA trend filter: rising HMA = bullish, falling HMA = bearish
        hma_rising = hma_1d_aligned[i] > hma_1d_aligned[i-1] if i > 0 and not np.isnan(hma_1d_aligned[i-1]) else False
        hma_falling = hma_1d_aligned[i] < hma_1d_aligned[i-1] if i > 0 and not np.isnan(hma_1d_aligned[i-1]) else False
        
        # Entry conditions with volume confirmation and trend filter
        long_entry = breakout_up and hma_rising and volume_confirmed[i] and position != 1
        short_entry = breakout_down and hma_falling and volume_confirmed[i] and position != -1
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long: Donchian(10) breakdown OR ATR stoploss
            if close[i] < donchian_10_lower_aligned[i-1]:
                exit_long = True
            elif i > 0 and not np.isnan(atr_14_aligned[i-1]):
                # Track entry price implicitly through position - use close-based stop
                # Simplified: exit if price drops 2*ATR from recent high (approximation)
                recent_high = np.maximum.reduce(high[max(0, i-10):i+1]) if i >= 10 else np.maximum.reduce(high[:i+1])
                if close[i] < recent_high - 2.0 * atr_14_aligned[i-1]:
                    exit_long = True
        
        elif position == -1:
            # Exit short: Donchian(10) breakout OR ATR stoploss
            if close[i] > donchian_10_upper_aligned[i-1]:
                exit_short = True
            elif i > 0 and not np.isnan(atr_14_aligned[i-1]):
                # Track entry price implicitly through position - use close-based stop
                # Simplified: exit if price rises 2*ATR from recent low (approximation)
                recent_low = np.minimum.reduce(low[max(0, i-10):i+1]) if i >= 10 else np.minimum.reduce(low[:i+1])
                if close[i] > recent_low + 2.0 * atr_14_aligned[i-1]:
                    exit_short = True
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_donchian_hma_volume_filter_v1"
timeframe = "4h"
leverage = 1.0