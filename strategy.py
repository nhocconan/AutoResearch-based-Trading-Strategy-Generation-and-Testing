#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R reversal with 1d trend filter and volume confirmation
# - Williams %R(14) measures overbought/oversold levels
# - Long when Williams %R crosses above -80 from below AND 1d close > 1d open (bullish daily) AND volume > 1.3x 20-period volume SMA
# - Short when Williams %R crosses below -20 from above AND 1d close < 1d open (bearish daily) AND volume > 1.3x 20-period volume SMA
# - Exit: Williams %R crosses opposite threshold (-20 for long, -80 for short) or ATR trailing stop (2.0x ATR)
# - Uses 1d for signal direction (trend bias) and 12h for precise entry timing
# - Session filter: 08-20 UTC to avoid low-volume Asian session noise
# - Position sizing: 0.25 discrete level to control drawdown and minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag while maintaining statistical significance
# - Williams %R works in both bull and bear markets by identifying reversal points at extremes

name = "12h_1d_williamsr_reversal_v1"
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
    if len(df_1d) < 14:  # Need min_periods for Williams %R
        return signals
    
    # Calculate 1d candle direction (bullish/bearish) for trend filter
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    # Bullish 1d candle: close > open
    bullish_1d = close_1d > open_1d
    bearish_1d = close_1d < open_1d
    # Align to 12h timeframe with proper delay (completed 1d bar only)
    bullish_1d_aligned = align_htf_to_ltf(prices, df_1d, bullish_1d)
    bearish_1d_aligned = align_htf_to_ltf(prices, df_1d, bearish_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate ATR(10) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Calculate 20-period volume SMA for confirmation
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Williams %R(14) on 12h timeframe
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - close) / (highest_high_14 - lowest_low_14) * -100
    
    # Track entry price for trailing stop
    entry_price = np.full(n, np.nan)
    
    for i in range(20, n):  # Start from 20 to have sufficient lookback
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is invalid
        if (np.isnan(atr[i]) or np.isnan(volume_sma_20[i]) or
            np.isnan(williams_r[i]) or np.isnan(bullish_1d_aligned[i]) or np.isnan(bearish_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 12h volume > 1.3x 20-period volume SMA
        vol_confirm = volume[i] > 1.3 * volume_sma_20[i]
        
        # Williams %R reversal signals
        williams_r_prev = williams_r[i-1] if i > 0 else williams_r[i]
        williams_r_curr = williams_r[i]
        
        # Bullish reversal: Williams %R crosses above -80 from below
        bullish_reversal = (williams_r_prev <= -80) and (williams_r_curr > -80)
        # Bearish reversal: Williams %R crosses below -20 from above
        bearish_reversal = (williams_r_prev >= -20) and (williams_r_curr < -20)
        
        if position == 0:  # Flat - look for entry
            # Long: Williams %R bullish reversal AND 1d bullish AND volume confirmation
            if bullish_reversal and bullish_1d_aligned[i] and vol_confirm:
                position = 1
                signals[i] = 0.25
                entry_price[i] = close[i]  # Record entry price for trailing stop
            # Short: Williams %R bearish reversal AND 1d bearish AND volume confirmation
            elif bearish_reversal and bearish_1d_aligned[i] and vol_confirm:
                position = -1
                signals[i] = -0.25
                entry_price[i] = close[i]  # Record entry price for trailing stop
            else:
                signals[i] = 0.0
                # Carry forward NaN values for tracking
                if i > 0:
                    entry_price[i] = entry_price[i-1]
        elif position == 1:  # Long position - look for exit
            # Update entry price tracking
            entry_price[i] = entry_price[i-1]
            
            # ATR trailing stop: exit if price drops 2.0*ATR below entry price
            trailing_stop = entry_price[i] - 2.0 * atr[i]
            
            # Exit conditions: trailing stop hit OR Williams %R crosses below -20 (overbought)
            exit_condition = (close[i] < trailing_stop) or (williams_r_curr < -20)
            
            if exit_condition:
                position = 0
                signals[i] = 0.0
                # Reset tracking
                entry_price[i] = np.nan
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            # Update entry price tracking
            entry_price[i] = entry_price[i-1]
            
            # ATR trailing stop: exit if price rises 2.0*ATR above entry price
            trailing_stop = entry_price[i] + 2.0 * atr[i]
            
            # Exit conditions: trailing stop hit OR Williams %R crosses above -80 (oversold)
            exit_condition = (close[i] > trailing_stop) or (williams_r_curr > -80)
            
            if exit_condition:
                position = 0
                signals[i] = 0.0
                # Reset tracking
                entry_price[i] = np.nan
            else:
                signals[i] = -0.25
    
    return signals