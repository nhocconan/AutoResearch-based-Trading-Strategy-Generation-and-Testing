#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Williams %R reversal with 1-week trend filter and volume confirmation
# - Long when Williams %R(14) crosses above -80 (oversold bounce) AND weekly close > weekly open (bullish weekly trend) AND volume > 1.3x 20-day volume SMA
# - Short when Williams %R(14) crosses below -20 (overbought rejection) AND weekly close < weekly open (bearish weekly trend) AND volume > 1.3x 20-day volume SMA
# - Exit: Williams %R crosses above -20 for longs or below -80 for shorts, or ATR trailing stop (2.0x ATR)
# - Uses 1d for signal generation and 1w for trend filter (proven to work in both bull and bear markets)
# - Session filter: 08-20 UTC to avoid low-volume Asian session noise
# - Position sizing: 0.25 discrete level to control drawdown and minimize fee churn
# - Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag while maintaining statistical significance
# - Williams %R is effective at identifying reversal points in ranging markets and catching trends early

name = "1d_1w_williamsr_reversal_v1"
timeframe = "1d"
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
    
    # Load 1w data ONCE before loop (MTF rule compliance)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return signals
    
    # Calculate Williams %R(14) on daily timeframe
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - close) / (highest_high_14 - lowest_low_14) * -100
    
    # Williams %R crossover signals
    williams_r_prev = np.roll(williams_r, 1)
    williams_r_prev[0] = np.nan
    williams_r_above_80 = (williams_r_prev <= -80) & (williams_r > -80)  # Cross above -80
    williams_r_below_20 = (williams_r_prev >= -20) & (williams_r < -20)  # Cross below -20
    
    # Calculate 1w trend filter (bullish/bearish weekly candle)
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    bullish_1w = close_1w > open_1w
    bearish_1w = close_1w < open_1w
    # Align to daily timeframe with proper delay (completed weekly bar only)
    bullish_1w_aligned = align_htf_to_ltf(prices, df_1w, bullish_1w)
    bearish_1w_aligned = align_htf_to_ltf(prices, df_1w, bearish_1w)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-day volume SMA for confirmation
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
        if (np.isnan(williams_r[i]) or np.isnan(williams_r_prev[i]) or
            np.isnan(atr[i]) or np.isnan(volume_sma_20[i]) or
            np.isnan(bullish_1w_aligned[i]) or np.isnan(bearish_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: daily volume > 1.3x 20-day volume SMA
        vol_confirm = volume[i] > 1.3 * volume_sma_20[i]
        
        if position == 0:  # Flat - look for entry
            # Long: Williams %R crosses above -80 AND weekly bullish AND volume confirmation
            if williams_r_above_80[i] and bullish_1w_aligned[i] and vol_confirm:
                position = 1
                signals[i] = 0.25
                highest_since_entry[i] = high[i]  # Initialize trailing stop
            # Short: Williams %R crosses below -20 AND weekly bearish AND volume confirmation
            elif williams_r_below_20[i] and bearish_1w_aligned[i] and vol_confirm:
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
            
            # Exit conditions: trailing stop hit OR Williams %R crosses above -20 (overbought)
            exit_condition = (close[i] < trailing_stop) or williams_r_above_80[i]
            
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
            
            # Exit conditions: trailing stop hit OR Williams %R crosses below -80 (oversold)
            exit_condition = (close[i] > trailing_stop) or williams_r_below_20[i]
            
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