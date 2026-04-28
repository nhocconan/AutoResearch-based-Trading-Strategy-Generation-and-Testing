#!/usr/bin/env python3
# Hypothesis: 6h RSI divergence with 1-day Bollinger Bands and volume confirmation.
# RSI divergence (bullish/bearish) signals potential reversals at Bollinger Band extremes.
# In strong trends (ADX > 25 from daily data), we filter trades to align with the trend direction.
# Volume confirmation ensures momentum behind the move.
# Designed for 6h timeframe to target 50-150 total trades over 4 years (12-37/year).
# Works in both bull and bear markets by combining reversal signals with trend filters.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def _rsi(close, period=14):
    """Calculate Relative Strength Index"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _bollinger_bands(close, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    return upper, lower, sma

def _adx(high, low, close, period=14):
    """Calculate Average Directional Index"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    plus_di_smooth = pd.Series(plus_dm).rolling(window=period, min_periods=period).mean().values
    minus_di_smooth = pd.Series(minus_dm).rolling(window=period, min_periods=period).mean().values
    
    # DI values
    plus_di = np.where(atr != 0, plus_di_smooth / atr * 100, 0)
    minus_di = np.where(atr != 0, minus_di_smooth / atr * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily Bollinger Bands (20,2)
    bb_upper_1d, bb_lower_1d, bb_middle_1d = _bollinger_bands(close_1d, 20, 2)
    
    # Daily ADX for trend filter
    adx_1d = _adx(high_1d, low_1d, close_1d, 14)
    
    # Align daily indicators to 6h timeframe
    bb_upper_1d_aligned = align_htf_to_ltf(prices, df_1d, bb_upper_1d)
    bb_lower_1d_aligned = align_htf_to_ltf(prices, df_1d, bb_lower_1d)
    bb_middle_1d_aligned = align_htf_to_ltf(prices, df_1d, bb_middle_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 6h RSI for divergence detection
    rsi_6h = _rsi(close, 14)
    
    # Volume filter: volume > 1.3x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bb_upper_1d_aligned[i]) or np.isnan(bb_lower_1d_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(rsi_6h[i]) or
            np.isnan(volume_ma[i]) or i < 14):  # RSI needs warmup
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_1d_aligned[i] > 25
        
        # RSI divergence detection
        # Bullish divergence: price makes lower low, RSI makes higher low
        # Bearish divergence: price makes higher high, RSI makes lower high
        lookback = 5  # Look back 5 periods for divergence
        
        if i >= lookback:
            # Price points
            price_now = close[i]
            price_then = close[i - lookback]
            
            # RSI points
            rsi_now = rsi_6h[i]
            rsi_then = rsi_6h[i - lookback]
            
            # Bullish divergence: lower price low but higher RSI low
            bullish_div = (price_now < price_then) and (rsi_now > rsi_then)
            
            # Bearish divergence: higher price high but lower RSI high
            bearish_div = (price_now > price_then) and (rsi_now < rsi_then)
        else:
            bullish_div = False
            bearish_div = False
        
        # Bollinger Band conditions
        price_near_upper = close[i] >= bb_upper_1d_aligned[i]
        price_near_lower = close[i] <= bb_lower_1d_aligned[i]
        
        # Entry conditions
        # Long: bullish divergence at lower BB + uptrend
        long_entry = bullish_div and price_near_lower and strong_trend and volume_filter[i]
        # Short: bearish divergence at upper BB + downtrend
        short_entry = bearish_div and price_near_upper and strong_trend and volume_filter[i]
        
        # Exit conditions: when price returns to middle band or divergence fails
        long_exit = close[i] >= bb_middle_1d_aligned[i] or not bullish_div
        short_exit = close[i] <= bb_middle_1d_aligned[i] or not bearish_div
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_RSIDivergence_BBands_ADXFilter_Volume"
timeframe = "6h"
leverage = 1.0