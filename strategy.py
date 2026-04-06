#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Three Bar Reversal pattern with 1d trend filter and volume confirmation.
# Three Bar Reversal: Bullish = 3 consecutive lower closes + close > prior close
#                     Bearish = 3 consecutive higher closes + close < prior close
# Trend filter: 1d EMA(50) slope > 0 = bull, < 0 = bear
# Only take long in bull trend, short in bear trend
# Volume > 1.3x 20-period average for confirmation
# Exit: Opposite signal or stop loss at 2*ATR
# Works in bull markets (catches pullbacks in uptrend) and bear markets (catches bounces in downtrend)

name = "exp_13582_12h_three_bar_reversal_1d_trend_vol_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
EMA_TREND_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.3
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1d data for trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = calculate_ema(close_1d, EMA_TREND_PERIOD)
    ema_1d_slope = np.diff(ema_1d, prepend=ema_1d[0])  # slope approximation
    ema_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_slope)
    
    # Calculate 12h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Three Bar Reversal components
    # Bullish: 3 consecutive lower closes + close > prior close
    close_down_1 = np.roll(close, 1) > close
    close_down_2 = np.roll(close, 2) > close
    close_down_3 = np.roll(close, 3) > close
    three_down = close_down_1 & close_down_2 & close_down_3
    close_up_prior = close > np.roll(close, 1)
    bullish_reversal = three_down & close_up_prior
    
    # Bearish: 3 consecutive higher closes + close < prior close
    close_up_1 = np.roll(close, 1) < close
    close_up_2 = np.roll(close, 2) < close
    close_up_3 = np.roll(close, 3) < close
    three_up = close_up_1 & close_up_2 & close_up_3
    close_down_prior = close < np.roll(close, 1)
    bearish_reversal = three_up & close_down_prior
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_TREND_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD, 3) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_1d_slope_aligned[i]) or np.isnan(bullish_reversal[i]) or np.isnan(bearish_reversal[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Trend from 1d EMA slope
        bull_trend = ema_1d_slope_aligned[i] > 0
        bear_trend = ema_1d_slope_aligned[i] < 0
        
        # Three Bar Reversal signals with trend filter
        long_signal = volume_ok and bullish_reversal[i] and bull_trend
        short_signal = volume_ok and bearish_reversal[i] and bear_trend
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on bearish reversal or trend change
            if bearish_reversal[i] or not bull_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Exit short on bullish reversal or trend change
            if bullish_reversal[i] or not bear_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals