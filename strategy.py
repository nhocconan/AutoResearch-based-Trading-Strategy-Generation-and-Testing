#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h TRIX momentum with 1d trend filter and volume confirmation
# - Long when TRIX crosses above zero AND 1d close > 1d SMA(50) AND volume > 1.3x 20-period volume SMA
# - Short when TRIX crosses below zero AND 1d close < 1d SMA(50) AND volume > 1.3x 20-period volume SMA
# - Exit: opposite TRIX cross OR ATR trailing stop (2.0x ATR)
# - Uses 1d for trend bias (above/below 50-day SMA) and 12h for precise momentum entry
# - Session filter: 08-20 UTC to avoid low-volume Asian session noise
# - Position sizing: 0.25 discrete level to control drawdown and minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag while maintaining statistical significance
# - TRIX is excellent at catching momentum shifts in both bull and bear markets, especially when combined with trend filter

name = "12h_1d_trix_momentum_v1"
timeframe = "12h"
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
    open_time = prices['open_time'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Calculate 1d SMA(50) for trend filter
    close_1d = df_1d['close'].values
    sma_50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    # Bullish 1d trend: close > SMA50
    bullish_1d = close_1d > sma_50_1d
    bearish_1d = close_1d < sma_50_1d
    # Align to 12h timeframe with proper delay (completed 1d bar only)
    bullish_1d_aligned = align_htf_to_ltf(prices, df_1d, bullish_1d)
    bearish_1d_aligned = align_htf_to_ltf(prices, df_1d, bearish_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate TRIX(12) on 12h timeframe
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12) - then % change
    ema1 = pd.Series(close).ewm(span=12, min_periods=12, adjust=False).mean()
    ema2 = ema1.ewm(span=12, min_periods=12, adjust=False).mean()
    ema3 = ema2.ewm(span=12, min_periods=12, adjust=False).mean()
    trix = pd.Series(ema3).pct_change() * 100  # Convert to percentage
    trix_values = trix.values
    trix_prev = np.roll(trix_values, 1)
    trix_prev[0] = np.nan
    # TRIX cross above/below zero
    trix_cross_up = (trix_values > 0) & (trix_prev <= 0)
    trix_cross_down = (trix_values < 0) & (trix_prev >= 0)
    
    # Calculate ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = np.nan  # First value has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period volume SMA for confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Track highest high since entry for trailing stop (long)
    # Track lowest low since entry for trailing stop (short)
    highest_since_entry = np.full(n, np.nan)
    lowest_since_entry = np.full(n, np.nan)
    
    for i in range(20, n):  # Start from 20 to have sufficient lookback
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is invalid
        if (np.isnan(atr[i]) or np.isnan(volume_sma_20[i]) or
            np.isnan(bullish_1d_aligned[i]) or np.isnan(bearish_1d_aligned[i]) or
            np.isnan(trix_values[i]) or np.isnan(trix_prev[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 12h volume > 1.3x 20-period volume SMA
        vol_confirm = volume[i] > 1.3 * volume_sma_20[i]
        
        if position == 0:  # Flat - look for entry
            # Long: TRIX crosses above zero AND 1d bullish trend AND volume confirmation
            if trix_cross_up[i] and bullish_1d_aligned[i] and vol_confirm:
                position = 1
                signals[i] = 0.25
                highest_since_entry[i] = high[i]  # Initialize trailing stop
            # Short: TRIX crosses below zero AND 1d bearish trend AND volume confirmation
            elif trix_cross_down[i] and bearish_1d_aligned[i] and vol_confirm:
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
            
            # Exit conditions: trailing stop hit OR opposite TRIX cross
            exit_condition = (close[i] < trailing_stop) or trix_cross_down[i]
            
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
            
            # Exit conditions: trailing stop hit OR opposite TRIX cross
            exit_condition = (close[i] > trailing_stop) or trix_cross_up[i]
            
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