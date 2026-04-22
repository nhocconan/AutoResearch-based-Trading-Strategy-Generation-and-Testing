#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily candlestick patterns combined with weekly trend filter and volume confirmation
# This strategy targets swing trading on daily timeframe with low trade frequency (~10-20 trades/year)
# Uses: Daily Engulfing candle pattern + Weekly EMA50 trend filter + Volume spike confirmation
# Designed to work in both bull and bear markets by following the weekly trend direction
# Engulfing patterns signal strong reversals, weekly EMA prevents counter-trend trades
# Volume spike confirms institutional participation

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for Engulfing patterns
    df_daily = get_htf_data(prices, '1d')
    daily_open = df_daily['open'].values
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    daily_volume = df_daily['volume'].values
    
    # Load weekly data once for trend filter
    df_weekly = get_htf_data(prices, '1w')
    weekly_close = df_weekly['close'].values
    
    # Calculate daily Engulfing patterns
    # Bullish engulfing: current green candle completely engulfs previous red candle
    # Bearish engulfing: current red candle completely engulfs previous green candle
    bullish_engulf = (
        (daily_close > daily_open) &  # current candle is green
        (daily_open < daily_close.shift(1)) &  # current open below previous close
        (daily_close > daily_open.shift(1)) &  # current close above previous open
        (daily_open < daily_close.shift(1)) &  # redundant but clear
        (daily_close > daily_open.shift(1))
    )
    
    bearish_engulf = (
        (daily_close < daily_open) &  # current candle is red
        (daily_open > daily_close.shift(1)) &  # current open above previous close
        (daily_close < daily_open.shift(1)) &  # current close below previous open
        (daily_open > daily_close.shift(1)) &  # redundant but clear
        (daily_close < daily_open.shift(1))
    )
    
    # Calculate weekly EMA50 for trend filter
    weekly_ema50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily patterns to lower timeframe
    bullish_engulf_aligned = align_htf_to_ltf(prices, df_daily, bullish_engulf.astype(float))
    bearish_engulf_aligned = align_htf_to_ltf(prices, df_daily, bearish_engulf.astype(float))
    
    # Align weekly EMA50 to lower timeframe
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema50)
    
    # Volume spike filter (20-day average on daily data)
    vol_ma_20 = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_daily, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any data is not ready
        if (np.isnan(bullish_engulf_aligned[i]) or 
            np.isnan(bearish_engulf_aligned[i]) or 
            np.isnan(weekly_ema50_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        bullish = bullish_engulf_aligned[i] > 0.5
        bearish = bearish_engulf_aligned[i] > 0.5
        weekly_ema = weekly_ema50_aligned[i]
        vol_ma = vol_ma_20_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-day average
        vol_spike = volume > 2.0 * vol_ma
        
        if position == 0:
            # Long: bullish engulfing + price above weekly EMA50 + volume spike
            if bullish and price > weekly_ema and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: bearish engulfing + price below weekly EMA50 + volume spike
            elif bearish and price < weekly_ema and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: opposite engulfing pattern or price crosses weekly EMA50
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on bearish engulfing or price crosses below weekly EMA50
                if bearish or price < weekly_ema:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on bullish engulfing or price crosses above weekly EMA50
                if bullish or price > weekly_ema:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "Daily_Engulfing_WeeklyEMA50_Volume"
timeframe = "1d"
leverage = 1.0