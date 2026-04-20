#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Candlestick Pattern with 1-Day Trend Filter and Volume Confirmation
# Uses engulfing patterns (bullish/bearish) combined with 1-day EMA trend filter
# In bull market (price > 1-day EMA50): trade bullish engulfing patterns
# In bear market (price < 1-day EMA50): trade bearish engulfing patterns
# Volume confirmation: require volume > 1.3x 20-period average
# Designed to capture reversal points with trend alignment
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 50-period EMA on daily timeframe for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Calculate bullish and bearish engulfing patterns
    # Bullish engulfing: current green candle engulfs previous red candle
    bullish_engulf = (close > open_price) & (open_price <= close) & \
                     (close > open_price) & (open_price < close) & \
                     (open_price <= close.shift(1)) & (close >= open_price.shift(1))
    
    # Bearish engulfing: current red candle engulfs previous green candle
    bearish_engulf = (close < open_price) & (open_price >= close) & \
                     (close < open_price) & (open_price > close) & \
                     (open_price >= close.shift(1)) & (close <= open_price.shift(1))
    
    # Fix the logic for engulfing patterns
    bullish_engulf = (close > open_price) & (open_price < close.shift(1)) & (close >= open_price.shift(1))
    bearish_engulf = (close < open_price) & (open_price > close.shift(1)) & (close <= open_price.shift(1))
    
    # Calculate volume filter: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market trend
        is_bull = close[i] > ema50_1d_aligned[i]
        is_bear = close[i] < ema50_1d_aligned[i]
        
        # Volume confirmation
        has_volume = vol_filter[i]
        
        if position == 0:
            # Enter long: bullish engulfing in bull market
            long_signal = False
            if has_volume and is_bull and bullish_engulf[i]:
                long_signal = True
            
            # Enter short: bearish engulfing in bear market
            short_signal = False
            if has_volume and is_bear and bearish_engulf[i]:
                short_signal = True
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish engulfing or trend reversal
            exit_signal = False
            if has_volume and bearish_engulf[i]:
                exit_signal = True
            elif close[i] < ema50_1d_aligned[i]:  # Trend turned bearish
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish engulfing or trend reversal
            exit_signal = False
            if has_volume and bullish_engulf[i]:
                exit_signal = True
            elif close[i] > ema50_1d_aligned[i]:  # Trend turned bullish
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Engulfing_TrendFilter_Volume"
timeframe = "12h"
leverage = 1.0