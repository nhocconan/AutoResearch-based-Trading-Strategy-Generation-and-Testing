#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Camarilla_R2S2_Breakout_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # === Weekly Candlestick Data ===
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_open = df_1w['open'].values
    
    # === Weekly Range for Candlestick Patterns ===
    weekly_range = weekly_high - weekly_low
    
    # === Weekly Bullish/Bearish Engulfing Detection ===
    # Bullish engulfing: current week bullish candle engulfs previous week bearish candle
    # Bearish engulfing: current week bearish candle engulfs previous week bullish candle
    prev_weekly_close = np.roll(weekly_close, 1)
    prev_weekly_open = np.roll(weekly_open, 1)
    prev_weekly_high = np.roll(weekly_high, 1)
    prev_weekly_low = np.roll(weekly_low, 1)
    
    # Set first values to avoid look-ahead
    prev_weekly_close[0] = weekly_close[0]
    prev_weekly_open[0] = weekly_open[0]
    prev_weekly_high[0] = weekly_high[0]
    prev_weekly_low[0] = weekly_low[0]
    
    # Bullish engulfing conditions
    bullish_engulfing = (
        (weekly_close > weekly_open) &  # Current week bullish
        (prev_weekly_close < prev_weekly_open) &  # Previous week bearish
        (weekly_open <= prev_weekly_close) &  # Current open <= previous close
        (weekly_close >= prev_weekly_open)   # Current close >= previous open
    )
    
    # Bearish engulfing conditions
    bearish_engulfing = (
        (weekly_close < weekly_open) &  # Current week bearish
        (prev_weekly_close > prev_weekly_open) &  # Previous week bullish
        (weekly_open >= prev_weekly_close) &  # Current open >= previous close
        (weekly_close <= prev_weekly_open)   # Current close <= previous open
    )
    
    # Align weekly patterns to daily timeframe
    bullish_engulfing_aligned = align_htf_to_ltf(prices, df_1w, bullish_engulfing.astype(float))
    bearish_engulfing_aligned = align_htf_to_ltf(prices, df_1w, bearish_engulfing.astype(float))
    
    # === Daily Volume Confirmation ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        close_val = prices['close'].iloc[i]
        vol_ratio_val = vol_ratio[i]
        bullish_engulfing_val = bullish_engulfing_aligned[i]
        bearish_engulfing_val = bearish_engulfing_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(vol_ratio_val) or 
            np.isnan(bullish_engulfing_val) or 
            np.isnan(bearish_engulfing_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bullish engulfing on weekly + volume confirmation
            if bullish_engulfing_val > 0.5 and vol_ratio_val > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: Bearish engulfing on weekly + volume confirmation
            elif bearish_engulfing_val > 0.5 and vol_ratio_val > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bearish engulfing signal or trend exhaustion
            if bearish_engulfing_val > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bullish engulfing signal or trend exhaustion
            if bullish_engulfing_val > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals