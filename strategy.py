#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Engulfing_Pattern_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 1d: Daily trend (close vs open) ===
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    daily_bullish = close_1d > open_1d  # True if bullish day
    daily_bearish = close_1d < open_1d  # True if bearish day
    
    # Align daily trend to 4h timeframe
    daily_bullish_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish.astype(float))
    daily_bearish_aligned = align_htf_to_ltf(prices, df_1d, daily_bearish.astype(float))
    
    # === 4h: Bullish and Bearish Engulfing patterns ===
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Bullish engulfing: current candle engulfs previous bearish candle
    bullish_engulf = (close > open_price) & (open_price < close) & \
                     (close > np.roll(open_price, 1)) & (open_price < np.roll(close, 1))
    # Bearish engulfing: current candle engulfs previous bullish candle
    bearish_engulf = (close < open_price) & (open_price > close) & \
                     (close < np.roll(open_price, 1)) & (open_price > np.roll(close, 1))
    
    # === 4h: Volume confirmation (current vs 20-period average) ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # === 4h: Trend filter using EMA 50 ===
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        close_val = close[i]
        open_val = open_price[i]
        bull_engulf = bullish_engulf[i]
        bear_engulf = bearish_engulf[i]
        vol_ratio_val = vol_ratio[i]
        ema_val = ema_50[i]
        daily_bull = daily_bullish_aligned[i]
        daily_bear = daily_bearish_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(vol_ratio_val) or np.isnan(ema_val) or 
            np.isnan(daily_bull) or np.isnan(daily_bear)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bullish engulfing + daily bullish bias + volume + price above EMA50
            if (bull_engulf and 
                daily_bull > 0.5 and 
                vol_ratio_val > 1.5 and 
                close_val > ema_val):
                signals[i] = 0.25
                position = 1
            # Short: Bearish engulfing + daily bearish bias + volume + price below EMA50
            elif (bear_engulf and 
                  daily_bear > 0.5 and 
                  vol_ratio_val > 1.5 and 
                  close_val < ema_val):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bearish engulfing or price below EMA50
            if bear_engulf or close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bullish engulfing or price above EMA50
            if bull_engulf or close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals