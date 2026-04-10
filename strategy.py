#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and volume confirmation
# - Long when price breaks above Camarilla H3 level AND 4h close > 4h open (bullish 4h candle) AND volume > 1.5x 20-period volume SMA
# - Short when price breaks below Camarilla L3 level AND 4h close < 4h open (bearish 4h candle) AND volume > 1.5x 20-period volume SMA
# - Exit: price reversion to Camarilla pivot point (mid-level) or ATR trailing stop (1.5x ATR)
# - Uses 4h for signal direction (trend bias) and 1h for precise entry timing
# - Session filter: 08-20 UTC to avoid low-volume Asian session noise
# - Position sizing: 0.20 discrete level to control drawdown and minimize fee churn
# - Target: 15-37 trades/year (60-150 total over 4 years) to minimize fee drag while maintaining statistical significance
# - Camarilla levels provide institutional support/resistance that works in both bull and bear markets
# - Uses previous completed 4h bar for Camarilla calculation to avoid look-ahead

name = "1h_4h_camarilla_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 4h data ONCE before loop (MTF rule compliance)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return signals
    
    # Calculate 4h candle direction (bullish/bearish) for trend filter
    close_4h = df_4h['close'].values
    open_4h = df_4h['open'].values
    # Bullish 4h candle: close > open
    bullish_4h = close_4h > open_4h
    bearish_4h = close_4h < open_4h
    # Align to 1h timeframe with proper delay (completed 4h bar only)
    bullish_4h_aligned = align_htf_to_ltf(prices, df_4h, bullish_4h)
    bearish_4h_aligned = align_htf_to_ltf(prices, df_4h, bearish_4h)
    
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
            np.isnan(bullish_4h_aligned[i]) or np.isnan(bearish_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1h volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > 1.5 * volume_sma_20[i]
        
        # Calculate Camarilla pivot levels for today (using previous day's OHLC)
        # Need to get previous day's high, low, close from 4h data
        if len(df_4h) >= 2:
            # Get previous day's OHLC (completed 4h bar)
            prev_high_4h = df_4h['high'].shift(1).values
            prev_low_4h = df_4h['low'].shift(1).values
            prev_close_4h = df_4h['close'].shift(1).values
            
            # Align previous day's OHLC to 1h timeframe
            prev_high_4h_aligned = align_htf_to_ltf(prices, df_4h, prev_high_4h)
            prev_low_4h_aligned = align_htf_to_ltf(prices, df_4h, prev_low_4h)
            prev_close_4h_aligned = align_htf_to_ltf(prices, df_4h, prev_close_4h)
            
            # Calculate Camarilla levels
            # H4 = Close + 1.5*(High-Low)
            # H3 = Close + 1.125*(High-Low)
            # H2 = Close + 0.75*(High-Low)
            # H1 = Close + 0.5*(High-Low)
            # Pivot = (High + Low + Close)/3
            # L1 = Close - 0.5*(High-Low)
            # L2 = Close - 0.75*(High-Low)
            # L3 = Close - 1.125*(High-Low)
            # L4 = Close - 1.5*(High-Low)
            
            rang = prev_high_4h_aligned - prev_low_4h_aligned
            camarilla_h3 = prev_close_4h_aligned + 1.125 * rang
            camarilla_l3 = prev_close_4h_aligned - 1.125 * rang
            camarilla_pivot = (prev_high_4h_aligned + prev_low_4h_aligned + prev_close_4h_aligned) / 3.0
            
            # Check for valid Camarilla levels
            if np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or np.isnan(camarilla_pivot[i]):
                signals[i] = 0.0
                continue
        else:
            signals[i] = 0.0
            continue
        
        # Camarilla breakout signals
        breakout_up = close[i] > camarilla_h3[i-1]  # Break above H3 level
        breakout_down = close[i] < camarilla_l3[i-1]  # Break below L3 level
        
        if position == 0:  # Flat - look for entry
            # Long: price breaks above H3 AND 4h bullish AND volume confirmation
            if breakout_up and bullish_4h_aligned[i] and vol_confirm:
                position = 1
                signals[i] = 0.20
                highest_since_entry[i] = high[i]  # Initialize trailing stop
            # Short: price breaks below L3 AND 4h bearish AND volume confirmation
            elif breakout_down and bearish_4h_aligned[i] and vol_confirm:
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
            
            # ATR trailing stop: exit if price drops 1.5*ATR below highest high since entry
            trailing_stop = highest_since_entry[i] - 1.5 * atr[i]
            
            # Exit conditions: trailing stop hit OR reversion to pivot point
            exit_condition = (close[i] < trailing_stop) or (close[i] < camarilla_pivot[i])
            
            if exit_condition:
                position = 0
                signals[i] = 0.0
                # Reset tracking arrays
                highest_since_entry[i] = np.nan
                lowest_since_entry[i] = np.nan
            else:
                signals[i] = 0.20
                # Propagate tracking values
                highest_since_entry[i] = highest_since_entry[i]
                lowest_since_entry[i] = lowest_since_entry[i-1]
        else:  # position == -1 (Short position) - look for exit
            # Update lowest low since entry
            lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
            
            # ATR trailing stop: exit if price rises 1.5*ATR above lowest low since entry
            trailing_stop = lowest_since_entry[i] + 1.5 * atr[i]
            
            # Exit conditions: trailing stop hit OR reversion to pivot point
            exit_condition = (close[i] > trailing_stop) or (close[i] > camarilla_pivot[i])
            
            if exit_condition:
                position = 0
                signals[i] = 0.0
                # Reset tracking arrays
                highest_since_entry[i] = np.nan
                lowest_since_entry[i] = np.nan
            else:
                signals[i] = -0.20
                # Propagate tracking values
                highest_since_entry[i] = highest_since_entry[i-1]
                lowest_since_entry[i] = lowest_since_entry[i]
    
    return signals