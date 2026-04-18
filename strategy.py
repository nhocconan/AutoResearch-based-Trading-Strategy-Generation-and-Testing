#!/usr/bin/env python3
"""
Strategy: 4h_OrderBlock_Breakout_Volume
Hypothesis: Institutional order blocks (OB) act as strong support/resistance. 
Breakouts from OB with volume confirmation and trend filter (EMA21) capture 
institutional flow. Works in bull/bear markets by trading breakouts in 
direction of higher-timeframe trend. Targets 20-40 trades/year to avoid 
fee drag.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for order block detection and EMA
    df_4h = get_htf_data(prices, '4h')
    
    # Order block: bullish OB = bearish candle before strong up move
    # Bearish OB = bullish candle before strong down move
    # Using 4h close/open for simplicity
    open_4h = df_4h['open'].values
    close_4h = df_4h['close'].values
    
    # Bullish OB: previous candle bearish (close < open) and current bullish
    # Bearish OB: previous candle bullish (close > open) and current bearish
    bullish_ob = (close_4h[:-1] < open_4h[:-1]) & (close_4h[1:] > open_4h[1:])
    bearish_ob = (close_4h[:-1] > open_4h[:-1]) & (close_4h[1:] < open_4h[1:])
    
    # OB levels: use the high/low of the OB candle
    # Bullish OB forms support at its low
    bullish_ob_low = np.concatenate([[np.nan], low_4h[:-1]])  # shift right
    # Bearish OB forms resistance at its high
    bearish_ob_high = np.concatenate([[np.nan], high_4h[:-1]])  # shift right
    
    # EMA21 for trend filter on 4h
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume average (20-period) on 4h
    vol_ma_20_4h = pd.Series(df_4h['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align to 4h timeframe (already 4h, but using for consistency)
    bullish_ob_low_4h = align_htf_to_ltf(prices, df_4h, bullish_ob_low)
    bearish_ob_high_4h = align_htf_to_ltf(prices, df_4h, bearish_ob_high)
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    # For 1h timeframe, we need to align 4h data to 1h
    # But since we're using 4h as primary, we'll use close prices directly
    # Actually, we need to work on the 4h timeframe signals
    
    # Let's switch approach: work directly on 4h data and generate signals per 4h bar
    # But the function expects same length as prices (which is 1h)
    
    # Revert: use 1h prices, get 4h data for signals
    
    # Recalculate with proper alignment
    
    # Get 1h data for entry timing
    # Actually, let's keep it simple: use 4h data but align to 1h index
    
    # Start over with clear approach
    
    # Use 1h prices for the array length
    # Get 4h data for signal generation
    
    # Reset and simplify
    
    # Get 4h data
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate indicators on 4h
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    vol_4h = df_4h['volume'].values
    
    # EMA21 trend
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume MA
    vol_ma_20_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    
    # Order blocks: simple definition
    # Bullish OB: when we have a down candle followed by up candle that closes above midpoint of down candle
    # Bearish OB: up candle followed by down candle that closes below midpoint of up candle
    
    # Initialize OB arrays
    bullish_ob = np.full_like(close_4h, np.nan)
    bearish_ob = np.full_like(close_4h, np.nan)
    
    for i in range(1, len(close_4h)):
        # Bullish OB: previous candle bearish, current bullish and closes above midpoint of prev
        if close_4h[i-1] < open_4h[i-1] and close_4h[i] > open_4h[i]:
            midpoint = (open_4h[i-1] + close_4h[i-1]) / 2
            if close_4h[i] > midpoint:
                bullish_ob[i] = low_4h[i-1]  # OB low is the low of the bearish candle
        
        # Bearish OB: previous candle bullish, current bearish and closes below midpoint of prev
        if close_4h[i-1] > open_4h[i-1] and close_4h[i] < open_4h[i]:
            midpoint = (open_4h[i-1] + close_4h[i-1]) / 2
            if close_4h[i] < midpoint:
                bearish_ob[i] = high_4h[i-1]  # OB high is the high of the bullish candle
    
    # Align 4h data to 1h timeframe
    ema_21_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    bullish_ob_aligned = align_htf_to_ltf(prices, df_4h, bullish_ob)
    bearish_ob_aligned = align_htf_to_ltf(prices, df_4h, bearish_ob)
    
    # Signals array
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 30  # need enough for EMA and OB
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(ema_21_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(bullish_ob_aligned[i]) or np.isnan(bearish_ob_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter
        uptrend = close[i] > ema_21_aligned[i]
        downtrend = close[i] < ema_21_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_20_aligned[i]
        
        # Breakout conditions
        # Long: price breaks above bearish OB (resistance) with volume in uptrend
        # Short: price breaks below bullish OB (support) with volume in downtrend
        breakout_long = close[i] > bearish_ob_aligned[i]
        breakdown_short = close[i] < bullish_ob_aligned[i]
        
        if position == 0:
            # Long: uptrend + volume + break above bearish OB
            if uptrend and vol_confirm and breakout_long:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + volume + break below bullish OB
            elif downtrend and vol_confirm and breakdown_short:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend breakdown or break below bullish OB (support)
            if not uptrend or breakdown_short:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend reversal or break above bearish OB (resistance)
            if not downtrend or breakout_long:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_OrderBlock_Breakout_Volume"
timeframe = "4h"
leverage = 1.0