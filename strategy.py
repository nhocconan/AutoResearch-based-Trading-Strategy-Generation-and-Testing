#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 12h regime filter
# - Bull Power = High - EMA(13); Bear Power = EMA(13) - Low
# - Long when Bull Power > 0 AND Bear Power < 0 AND 12h close > 12h EMA(50) (bullish regime)
# - Short when Bear Power > 0 AND Bull Power < 0 AND 12h close < 12h EMA(50) (bearish regime)
# - Exit: opposite signal or ATR(10) trailing stop (2.0x ATR)
# - Uses 12h for regime filter (trend bias) and 6h for Elder Ray signals
# - Session filter: 08-20 UTC to avoid low-volume Asian session noise
# - Position sizing: 0.25 discrete level to control drawdown and minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag while maintaining statistical significance
# - Elder Ray measures bull/bear strength behind price moves; regime filter ensures trading with higher timeframe trend

name = "6h_12h_elder_ray_regime_v1"
timeframe = "6h"
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
    
    # Load 12h data ONCE before loop (MTF rule compliance)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return signals
    
    # Calculate 12h EMA(50) for regime filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align to 6h timeframe with proper delay (completed 12h bar only)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate EMA(13) for Elder Ray on 6h
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA
    bear_power = ema_13 - low   # Bear Power = EMA - Low
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate ATR(10) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Track highest high since entry for trailing stop (long)
    # Track lowest low since entry for trailing stop (short)
    highest_since_entry = np.full(n, np.nan)
    lowest_since_entry = np.full(n, np.nan)
    
    for i in range(13, n):  # Start from 13 to have sufficient lookback for EMA(13)
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is invalid
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: 12h close above/below EMA(50)
        bullish_regime = close_12h[i] > ema_50_12h[i]  # Using current 12h close (already aligned)
        bearish_regime = close_12h[i] < ema_50_12h[i]
        
        # Elder Ray signals
        bull_power_pos = bull_power[i] > 0
        bear_power_pos = bear_power[i] > 0
        
        if position == 0:  # Flat - look for entry
            # Long: Bull Power > 0 AND Bear Power < 0 AND bullish regime
            if bull_power_pos and (not bear_power_pos) and bullish_regime:
                position = 1
                signals[i] = 0.25
                highest_since_entry[i] = high[i]  # Initialize trailing stop
            # Short: Bear Power > 0 AND Bull Power < 0 AND bearish regime
            elif bear_power_pos and (not bull_power_pos) and bearish_regime:
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
            
            # Exit conditions: trailing stop hit OR opposite Elder Ray signal
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
            
            # Exit conditions: trailing stop hit OR opposite Elder Ray signal
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