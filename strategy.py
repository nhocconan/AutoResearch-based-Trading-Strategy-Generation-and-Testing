#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and ATR trailing stop
# - Long when price breaks above 20-day Donchian high AND 1w close > 1w open (bullish weekly candle)
# - Short when price breaks below 20-day Donchian low AND 1w close < 1w open (bearish weekly candle)
# - Exit: ATR(14) trailing stop (2.5x ATR from extreme) OR Donchian mid-line reversion
# - Uses 1w for trend bias (avoids whipsaw in bear markets) and 1d for precise entries
# - Position sizing: 0.25 discrete level to control drawdown and minimize fee churn
# - Target: 7-25 trades/year (30-100 total over 4 years) to minimize fee drag
# - Donchian breakouts work in both bull and bear markets when filtered by higher timeframe trend

name = "1d_1w_donchian_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1w data ONCE before loop (MTF rule compliance)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return signals
    
    # Calculate 1w candle direction (bullish/bearish) for trend filter
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    # Bullish 1w candle: close > open
    bullish_1w = close_1w > open_1w
    bearish_1w = close_1w < open_1w
    # Align to 1d timeframe with proper delay (completed 1w bar only)
    bullish_1w_aligned = align_htf_to_ltf(prices, df_1w, bullish_1w)
    bearish_1w_aligned = align_htf_to_ltf(prices, df_1w, bearish_1w)
    
    # Calculate ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Track highest high since entry for trailing stop (long)
    # Track lowest low since entry for trailing stop (short)
    highest_since_entry = np.full(n, np.nan)
    lowest_since_entry = np.full(n, np.nan)
    
    for i in range(20, n):  # Start from 20 to have sufficient lookback for Donchian
        
        # Skip if any required data is invalid
        if (np.isnan(atr[i]) or
            np.isnan(bullish_1w_aligned[i]) or np.isnan(bearish_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Calculate Donchian channels (20-period) using pure numpy/pandas
        # Highest high of last 20 periods (excluding current)
        highest_high = np.max(high[i-20:i]) if i >= 20 else np.nan
        # Lowest low of last 20 periods (excluding current)
        lowest_low = np.min(low[i-20:i]) if i >= 20 else np.nan
        
        if np.isnan(highest_high) or np.isnan(lowest_low):
            signals[i] = 0.0
            continue
        
        # Donchian mid-line
        donchian_mid = (highest_high + lowest_low) / 2.0
        
        # Donchian breakout signals
        breakout_up = close[i] > highest_high  # Break above 20-period high
        breakout_down = close[i] < lowest_low  # Break below 20-period low
        
        if position == 0:  # Flat - look for entry
            # Long: price breaks above Donchian high AND 1w bullish
            if breakout_up and bullish_1w_aligned[i]:
                position = 1
                signals[i] = 0.25
                highest_since_entry[i] = high[i]  # Initialize trailing stop
            # Short: price breaks below Donchian low AND 1w bearish
            elif breakout_down and bearish_1w_aligned[i]:
                position = -1
                signals[i] = -0.25
                lowest_since_entry[i] = low[i]  # Initialize trailing stop
            else:
                signals[i] = 0.0
                # Carry forward NaN values for tracking
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
        elif position == 1:  # Long position - look for exit
            # Update highest high since entry
            highest_since_entry[i] = max(highest_since_entry[i-1], high[i])
            
            # ATR trailing stop: exit if price drops 2.5*ATR below highest high since entry
            trailing_stop = highest_since_entry[i] - 2.5 * atr[i]
            
            # Exit conditions: trailing stop hit OR reversion to Donchian mid-line
            exit_condition = (close[i] < trailing_stop) or (close[i] < donchian_mid)
            
            if exit_condition:
                position = 0
                signals[i] = 0.0
                # Reset tracking arrays
                highest_since_entry[i] = np.nan
                lowest_since_entry[i] = np.nan
            else:
                signals[i] = 0.25
                # Propagate tracking values
                highest_since_entry[i] = highest_since_entry[i]
                lowest_since_entry[i] = lowest_since_entry[i-1]
        else:  # position == -1 (Short position) - look for exit
            # Update lowest low since entry
            lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
            
            # ATR trailing stop: exit if price rises 2.5*ATR above lowest low since entry
            trailing_stop = lowest_since_entry[i] + 2.5 * atr[i]
            
            # Exit conditions: trailing stop hit OR reversion to Donchian mid-line
            exit_condition = (close[i] > trailing_stop) or (close[i] > donchian_mid)
            
            if exit_condition:
                position = 0
                signals[i] = 0.0
                # Reset tracking arrays
                highest_since_entry[i] = np.nan
                lowest_since_entry[i] = np.nan
            else:
                signals[i] = -0.25
                # Propagate tracking values
                highest_since_entry[i] = highest_since_entry[i-1]
                lowest_since_entry[i] = lowest_since_entry[i]
    
    return signals