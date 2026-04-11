#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d HMA trend filter and ATR-based stoploss
# - Long: Price breaks above Donchian upper channel (20-period high) + 1d HMA(21) rising
# - Short: Price breaks below Donchian lower channel (20-period low) + 1d HMA(21) falling
# - Exit: ATR-based trailing stop (2.0 ATR from extreme) or opposite Donchian breakout
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 19-50 trades/year (75-200 total over 4 years) to stay within fee drag limits
# - Donchian channels provide clear structure for breakouts in both bull and bear markets
# - 1d HMA filter ensures we only trade with the higher timeframe trend, reducing whipsaws

name = "4h_1d_donchian_breakout_hma_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    
    # Load 1d data ONCE before loop for HMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute 1d HMA(21) for trend filter
    close_1d = df_1d['close'].values
    # HMA calculation: WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    # Calculate HMA manually for 1d
    if len(close_1d) >= 21:
        wma_half = np.array([wma(close_1d[i:i+half_len], half_len) if i+half_len <= len(close_1d) else np.nan 
                            for i in range(len(close_1d))])
        wma_full = np.array([wma(close_1d[i:i+21], 21) if i+21 <= len(close_1d) else np.nan 
                            for i in range(len(close_1d))])
        raw_hma = 2 * wma_half - wma_full
        hma_21 = np.array([wma(raw_hma[i:i+sqrt_len], sqrt_len) if i+sqrt_len <= len(raw_hma) else np.nan 
                          for i in range(len(raw_hma))])
        # Pad beginning with NaN
        hma_21 = np.concatenate([np.full(len(close_1d) - len(hma_21), np.nan), hma_21])
    else:
        hma_21 = np.full(len(close_1d), np.nan)
    
    hma_21_aligned = align_htf_to_ltf(prices, df_1d, hma_21)
    hma_rising = hma_21_aligned > np.roll(hma_21_aligned, 1)
    hma_falling = hma_21_aligned < np.roll(hma_21_aligned, 1)
    
    # Pre-compute Donchian channels on 4h timeframe
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute ATR for stoploss (4h timeframe)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    for i in range(50, n):  # Start after 50-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(hma_21_aligned[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        
        # Donchian levels
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price closes above upper Donchian channel + 1d HMA rising
        if close_price > upper_channel and hma_rising[i]:
            enter_long = True
        
        # Short breakout: price closes below lower Donchian channel + 1d HMA falling
        if close_price < lower_channel and hma_falling[i]:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price hits ATR stoploss or breaks below lower channel
            exit_long = (close_price <= long_stop) or (close_price < lower_channel)
        elif position == -1:
            # Exit short if price hits ATR stoploss or breaks above upper channel
            exit_short = (close_price >= short_stop) or (close_price > upper_channel)
        
        # Update stoploss levels when entering a position
        if enter_long:
            entry_price = close_price
            long_stop = entry_price - 2.0 * atr_14[i]
        elif enter_short:
            entry_price = close_price
            short_stop = entry_price + 2.0 * atr_14[i]
        
        # Update trailing stoploss for existing positions
        if position == 1:
            # Trail long stop upward: max of current stop and (high - 2*ATR)
            long_stop = max(long_stop, high[i] - 2.0 * atr_14[i])
        elif position == -1:
            # Trail short stop downward: min of current stop and (low + 2*ATR)
            short_stop = min(short_stop, low[i] + 2.0 * atr_14[i])
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals