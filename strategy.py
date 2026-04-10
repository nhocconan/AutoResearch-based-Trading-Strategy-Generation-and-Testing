#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1w/1d regime filter
# - Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# - Long when Bull Power > 0 AND Bear Power < 0 AND 1w close > 1w open (bullish weekly candle) AND 1d close > 1d open (bullish daily candle)
# - Short when Bear Power > 0 AND Bull Power < 0 AND 1w close < 1w open (bearish weekly candle) AND 1d close < 1d open (bearish daily candle)
# - Exit: opposite signal or ATR(14) trailing stop (2.0x ATR)
# - Uses 1w/1d for regime (trend bias) and 6h for Elder Ray calculation and precise entry timing
# - Session filter: 08-20 UTC to avoid low-volume Asian session noise
# - Position sizing: 0.25 discrete level to control drawdown and minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag while maintaining statistical significance
# - Elder Ray measures bull/bear power relative to EMA, effective in both trending and ranging markets when combined with HTF regime filter

name = "6h_1w_1d_elder_ray_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1w and 1d data ONCE before loop (MTF rule compliance)
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 2 or len(df_1d) < 2:
        return signals
    
    # Calculate 1w candle direction (bullish/bearish) for regime filter
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    bullish_1w = close_1w > open_1w
    bearish_1w = close_1w < open_1w
    # Align to 6h timeframe with proper delay (completed 1w bar only)
    bullish_1w_aligned = align_htf_to_ltf(prices, df_1w, bullish_1w)
    bearish_1w_aligned = align_htf_to_ltf(prices, df_1w, bearish_1w)
    
    # Calculate 1d candle direction (bullish/bearish) for regime filter
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    bullish_1d = close_1d > open_1d
    bearish_1d = close_1d < open_1d
    # Align to 6h timeframe with proper delay (completed 1d bar only)
    bullish_1d_aligned = align_htf_to_ltf(prices, df_1d, bullish_1d)
    bearish_1d_aligned = align_htf_to_ltf(prices, df_1d, bearish_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate EMA(13) for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema13  # Bull Power = High - EMA(13)
    bear_power = ema13 - low   # Bear Power = EMA(13) - Low
    
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
    
    for i in range(13, n):  # Start from 13 to have sufficient lookback for EMA
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is invalid
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(atr[i]) or np.isnan(bullish_1w_aligned[i]) or np.isnan(bearish_1w_aligned[i]) or
            np.isnan(bullish_1d_aligned[i]) or np.isnan(bearish_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Elder Ray signals
        bull_power_pos = bull_power[i] > 0
        bear_power_pos = bear_power[i] > 0
        
        if position == 0:  # Flat - look for entry
            # Long: Bull Power > 0 AND Bear Power < 0 AND 1w bullish AND 1d bullish
            if bull_power_pos and (not bear_power_pos) and bullish_1w_aligned[i] and bullish_1d_aligned[i]:
                position = 1
                signals[i] = 0.25
                highest_since_entry[i] = high[i]  # Initialize trailing stop
            # Short: Bear Power > 0 AND Bull Power < 0 AND 1w bearish AND 1d bearish
            elif bear_power_pos and (not bull_power_pos) and bearish_1w_aligned[i] and bearish_1d_aligned[i]:
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
            
            # ATR trailing stop: exit if price drops 2.0*ATR below highest high since entry
            trailing_stop = highest_since_entry[i] - 2.0 * atr[i]
            
            # Exit conditions: trailing stop hit OR opposite Elder Ray signal (Bear Power > 0 AND Bull Power < 0)
            exit_condition = (close[i] < trailing_stop) or (bear_power_pos and (not bull_power_pos))
            
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
            
            # ATR trailing stop: exit if price rises 2.0*ATR above lowest low since entry
            trailing_stop = lowest_since_entry[i] + 2.0 * atr[i]
            
            # Exit conditions: trailing stop hit OR opposite Elder Ray signal (Bull Power > 0 AND Bear Power < 0)
            exit_condition = (close[i] > trailing_stop) or (bull_power_pos and (not bear_power_pos))
            
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