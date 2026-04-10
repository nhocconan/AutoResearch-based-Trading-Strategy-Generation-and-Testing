#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h ATR-based breakout with 4h trend filter and daily regime filter
# - Long when price breaks above ATR(14) upper band AND 4h EMA21 > EMA50 (bullish 4h trend) AND daily chop < 61.8 (trending regime)
# - Short when price breaks below ATR(14) lower band AND 4h EMA21 < EMA50 (bearish 4h trend) AND daily chop < 61.8 (trending regime)
# - Exit: ATR trailing stop (2.0x ATR) or time-based exit (max 24 bars)
# - Uses 4h for trend direction (EMA crossover) and daily for regime filter (choppiness index)
# - 1h only for precise entry timing via ATR bands
# - Session filter: 08-20 UTC to avoid low-volume Asian session noise
# - Position sizing: 0.20 discrete level to control drawdown and minimize fee churn
# - Target: 15-37 trades/year (60-150 total over 4 years) to minimize fee drag while maintaining statistical significance
# - ATR bands provide volatility-adjusted breakout levels that adapt to market conditions
# - Choppiness index filter ensures we only trade in trending markets, avoiding choppy ranging conditions
# - Works in both bull and bear markets by following the 4h trend direction

name = "1h_4h_1d_atr_breakout_v1"
timeframe = "1h"
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
    bars_in_trade = np.zeros(n, dtype=int)  # Track bars in current trade for time-based exit
    
    # Load 4h data ONCE before loop (MTF rule compliance)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return signals
    
    # Load 1d data ONCE before loop (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Calculate 4h EMA21 and EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema21_4h = pd.Series(close_4h).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    # Bullish 4h trend: EMA21 > EMA50
    bullish_4h_trend = ema21_4h > ema50_4h
    bearish_4h_trend = ema21_4h < ema50_4h
    # Align to 1h timeframe with proper delay (completed 4h bar only)
    bullish_4h_aligned = align_htf_to_ltf(prices, df_4h, bullish_4h_trend)
    bearish_4h_aligned = align_htf_to_ltf(prices, df_4h, bearish_4h_trend)
    
    # Calculate daily choppiness index for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for daily
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    
    # ATR(14) for daily
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Sum of TRUE RANGE over 14 periods (numerator for chop)
    sum_tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Max(high) - Min(low) over 14 periods (denominator for chop)
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    
    # Choppiness Index: 100 * log10(sum_tr_14 / range_14) / log10(14)
    # Avoid division by zero and log of zero
    with np.errstate(divide='ignore', invalid='ignore'):
        chop_ratio = np.where(range_14 > 0, sum_tr_14 / range_14, 1.0)
        chop_ratio = np.where(chop_ratio > 0, chop_ratio, 1.0)
        chop = 100 * np.log10(chop_ratio) / np.log10(14)
    
    # Trending regime: chop < 61.8 (below this = trending, above = ranging)
    trending_regime = chop < 61.8
    # Align to 1h timeframe with proper delay (completed 1d bar only)
    trending_regime_aligned = align_htf_to_ltf(prices, df_1d, trending_regime)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate ATR(14) for 1h timeframe (for bands and trailing stop)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR bands: ±1.5 * ATR from close
    atr_multiplier = 1.5
    upper_band = close + atr_multiplier * atr
    lower_band = close - atr_multiplier * atr
    
    # Track highest high since entry for trailing stop (long)
    # Track lowest low since entry for trailing stop (short)
    highest_since_entry = np.full(n, np.nan)
    lowest_since_entry = np.full(n, np.nan)
    
    for i in range(20, n):  # Start from 20 to have sufficient lookback
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            bars_in_trade[i] = 0
            continue
            
        # Skip if any required data is invalid
        if (np.isnan(atr[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(bullish_4h_aligned[i]) or np.isnan(bearish_4h_aligned[i]) or
            np.isnan(trending_regime_aligned[i])):
            signals[i] = 0.0
            bars_in_trade[i] = 0
            continue
        
        # Update bars in trade counter
        if position != 0:
            bars_in_trade[i] = bars_in_trade[i-1] + 1
        else:
            bars_in_trade[i] = 0
        
        # Breakout conditions
        breakout_up = close[i] > upper_band[i-1]  # Break above upper band
        breakout_down = close[i] < lower_band[i-1]  # Break below lower band
        
        if position == 0:  # Flat - look for entry
            # Long: price breaks above upper band AND 4h bullish trend AND trending regime
            if breakout_up and bullish_4h_aligned[i] and trending_regime_aligned[i]:
                position = 1
                signals[i] = 0.20
                highest_since_entry[i] = high[i]  # Initialize trailing stop
            # Short: price breaks below lower band AND 4h bearish trend AND trending regime
            elif breakout_down and bearish_4h_aligned[i] and trending_regime_aligned[i]:
                position = -1
                signals[i] = -0.20
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
            
            # Exit conditions: trailing stop hit OR time-based exit (max 24 bars) OR break below lower band
            exit_condition = (close[i] < trailing_stop) or (bars_in_trade[i] >= 24) or (close[i] < lower_band[i])
            
            if exit_condition:
                position = 0
                signals[i] = 0.0
                # Reset tracking arrays
                highest_since_entry[i] = np.nan
                lowest_since_entry[i] = np.nan
                bars_in_trade[i] = 0
            else:
                signals[i] = 0.20
                # Propagate tracking values
                highest_since_entry[i] = highest_since_entry[i]
                lowest_since_entry[i] = lowest_since_entry[i-1]
        else:  # position == -1 (Short position) - look for exit
            # Update lowest low since entry
            lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
            
            # ATR trailing stop: exit if price rises 2.0*ATR above lowest low since entry
            trailing_stop = lowest_since_entry[i] + 2.0 * atr[i]
            
            # Exit conditions: trailing stop hit OR time-based exit (max 24 bars) OR break above upper band
            exit_condition = (close[i] > trailing_stop) or (bars_in_trade[i] >= 24) or (close[i] > upper_band[i])
            
            if exit_condition:
                position = 0
                signals[i] = 0.0
                # Reset tracking arrays
                highest_since_entry[i] = np.nan
                lowest_since_entry[i] = np.nan
                bars_in_trade[i] = 0
            else:
                signals[i] = -0.20
                # Propagate tracking values
                highest_since_entry[i] = highest_since_entry[i-1]
                lowest_since_entry[i] = lowest_since_entry[i]
    
    return signals