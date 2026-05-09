#!/usr/bin/env python3
# Hypothesis: 12h Candlestick Pattern + 1d Trend + Volume Spike
# Long when bullish engulfing pattern forms above 1d EMA34 and volume > 2x 20-period average
# Short when bearish engulfing pattern forms below 1d EMA34 and volume > 2x 20-period average
# Exit when opposite engulfing pattern forms or price crosses 1d EMA34 in opposite direction
# Position size: 0.25 to limit drawdown and reduce churn
# Designed to capture reversals in trending markets with confirmation, avoiding false signals

name = "12h_Engulfing_1dEMA_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h EMA20 for short-term trend (optional filter, not used directly in entry)
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Align 1d EMA34 to 12h timeframe (waits for daily close)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (2.0 * vol_ma.values)
    
    # Bullish engulfing: current bullish candle engulfs previous bearish candle
    bullish_engulfing = (close > open_price) & (open_price < close) & \
                        (close > open_price) & (open_price < close) & \
                        (close > open_price) & (open_price < close)  # Placeholder - will fix below
    
    # Correct bullish engulfing: current green candle fully engulfs previous red candle
    bullish_engulfing = (close > open_price) & (open_price < close) & \
                        (close > open_price) & (open_price < close)  # Still wrong
    
    # Proper bullish engulfing definition:
    bullish_engulfing = (close > open_price) & (open_price < close)  # Temp fix
    
    # Correct implementation:
    bullish_engulfing = (close > open_price) & (open_price < close)  # Still incorrect
    
    # Actual bullish engulling: current candle is bullish (close > open) and
    # its body completely engulfs the previous candle's body
    bullish_engulfing = (close > open_price) & \
                        (open_price < close) & \
                        (close > open_price) & \
                        (open_price < close)  # Giving up - using correct logic below
    
    # Proper bullish engulfing:
    bullish_engulfing = (close > open_price) & \
                        (open_price[1:] < close[:-1]) & \
                        (close[:-1] < open_price[:-1])  # No, this is not right either
    
    # Let's do it properly:
    # Bullish engulfing: current candle bullish (close > open) AND
    # current open <= previous close AND current close >= previous open
    bullish_engulfing = np.zeros(n, dtype=bool)
    bullish_engulfing[1:] = (close[1:] > open_price[1:]) & \
                            (open_price[1:] <= close[:-1]) & \
                            (close[1:] >= open_price[:-1])
    
    # Bearish engulfing: current candle bearish (close < open) AND
    # current open >= previous close AND current close <= previous open
    bearish_engulfing = np.zeros(n, dtype=bool)
    bearish_engulfing[1:] = (close[1:] < open_price[1:]) & \
                            (open_price[1:] >= close[:-1]) & \
                            (close[1:] <= open_price[:-1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: bullish engulfing, price above 1d EMA34, volume spike
            if (bullish_engulfing[i] and 
                close[i] > ema34_1d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: bearish engulfing, price below 1d EMA34, volume spike
            elif (bearish_engulfing[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish engulfing forms OR price crosses below 1d EMA34
            if bearish_engulfing[i] or (close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish engulfing forms OR price crosses above 1d EMA34
            if bullish_engulfing[i] or (close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals