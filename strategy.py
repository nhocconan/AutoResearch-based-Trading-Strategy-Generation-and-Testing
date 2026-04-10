#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R reversal with 1d trend filter and volume confirmation
# - Long when Williams %R crosses above -80 (oversold) AND 1d close > 1d SMA50 (bullish trend) AND volume > 1.3x 20-period volume SMA
# - Short when Williams %R crosses below -20 (overbought) AND 1d close < 1d SMA50 (bearish trend) AND volume > 1.3x 20-period volume SMA
# - Exit: Williams %R crosses below -50 (for long) or above -50 (for short) OR ATR trailing stop (2.0x ATR)
# - Uses 1d SMA50 for trend bias and 4h Williams %R for precise reversal timing
# - Session filter: 08-20 UTC to avoid low-volume Asian session noise
# - Position sizing: 0.25 discrete level to control drawdown and minimize fee churn
# - Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag while maintaining statistical significance
# - Williams %R is effective at identifying exhaustion points in both bull and bear markets, especially when combined with trend filter

name = "4h_1d_williamsr_reversal_v1"
timeframe = "4h"
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
    
    # Load 1d data ONCE before loop (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Calculate 1d SMA50 for trend filter
    close_1d = df_1d['close'].values
    sma50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    bullish_trend = close_1d > sma50_1d
    bearish_trend = close_1d < sma50_1d
    # Align to 4h timeframe with proper delay (completed 1d bar only)
    bullish_trend_aligned = align_htf_to_ltf(prices, df_1d, bullish_trend)
    bearish_trend_aligned = align_htf_to_ltf(prices, df_1d, bearish_trend)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period volume SMA for confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Williams %R(14) for 4h
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
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
            np.isnan(bullish_trend_aligned[i]) or np.isnan(bearish_trend_aligned[i]) or
            np.isnan(williams_r[i]) or np.isnan(highest_high_14[i]) or np.isnan(lowest_low_14[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 4h volume > 1.3x 20-period volume SMA
        vol_confirm = volume[i] > 1.3 * volume_sma_20[i]
        
        # Williams %R signals: cross above -80 (long) or below -20 (short)
        cross_up_80 = (williams_r[i-1] <= -80) and (williams_r[i] > -80)
        cross_down_20 = (williams_r[i-1] >= -20) and (williams_r[i] < -20)
        cross_down_50 = (williams_r[i-1] > -50) and (williams_r[i] <= -50)  # exit long
        cross_up_50 = (williams_r[i-1] < -50) and (williams_r[i] >= -50)   # exit short
        
        if position == 0:  # Flat - look for entry
            # Long: Williams %R crosses above -80 AND bullish 1d trend AND volume confirmation
            if cross_up_80 and bullish_trend_aligned[i] and vol_confirm:
                position = 1
                signals[i] = 0.25
                highest_since_entry[i] = high[i]  # Initialize trailing stop
            # Short: Williams %R crosses below -20 AND bearish 1d trend AND volume confirmation
            elif cross_down_20 and bearish_trend_aligned[i] and vol_confirm:
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
            
            # Exit conditions: trailing stop hit OR Williams %R crosses below -50
            exit_condition = (close[i] < trailing_stop) or cross_down_50
            
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
            
            # Exit conditions: trailing stop hit OR Williams %R crosses above -50
            exit_condition = (close[i] > trailing_stop) or cross_up_50
            
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