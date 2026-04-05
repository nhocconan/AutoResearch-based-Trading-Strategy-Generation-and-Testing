#!/usr/bin/env python3
"""
Experiment #8359: 6-hour Bollinger Band squeeze breakout with 12-hour trend filter and volume confirmation.
Hypothesis: Bollinger Band width contraction indicates low volatility and impending breakout. 
Breakouts in the direction of the 12-hour trend (price above/below 50-period EMA) with volume 
>1.5x 20-period MA capture high-probability moves while avoiding whipsaw in ranging markets.
Targeting 50-150 total trades over 4 years (12-37/year) for optimal balance.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8359_6b_bollinger_squeeze_12h_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
BB_PERIOD = 20
BB_STD_DEV = 2.0
BBW_PERCENTILE_LOOKBACK = 50  # for squeeze detection
EMA_TREND_PERIOD = 50
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
ATR_TARGET_MULTIPLIER = 3.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=EMA_TREND_PERIOD, adjust=False, min_periods=EMA_TREND_PERIOD).mean().values
    # Trend: 1 = bullish (price above EMA), -1 = bearish (price below EMA)
    trend_12h = np.where(close_12h > ema_12h, 1, -1)
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands on 6h
    sma = pd.Series(close).rolling(window=BB_PERIOD, min_periods=BB_PERIOD).mean().values
    std = pd.Series(close).rolling(window=BB_PERIOD, min_periods=BB_PERIOD).std().values
    upper_band = sma + (BB_STD_DEV * std)
    lower_band = sma - (BB_STD_DEV * std)
    bb_width = upper_band - lower_band
    
    # Bollinger Band Width percentile for squeeze detection
    # Using pandas rolling percentile rank
    bbw_series = pd.Series(bb_width)
    bbw_percentile = bbw_series.rolling(window=BBW_PERCENTILE_LOOKBACK, min_periods=BBW_PERCENTILE_LOOKBACK).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    squeeze = bbw_percentile <= 0.2  # Bottom 20% indicates squeeze
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    
    # Start from warmup period
    start = max(BB_PERIOD, EMA_TREND_PERIOD, BBW_PERCENTILE_LOOKBACK, 
                VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(trend_12h_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss or target
        if position == 1:  # long position
            if close[i] <= stop_price or close[i] >= target_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price or close[i] <= target_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine trend bias from 12h EMA
        bull_bias = trend_12h_aligned[i] == 1   # 12h price above EMA
        bear_bias = trend_12h_aligned[i] == -1  # 12h price below EMA
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Breakout conditions - price closes beyond Bollinger Bands
        breakout_up = close[i] > upper_band[i-1] if i-1 >= 0 else False
        breakout_down = close[i] < lower_band[i-1] if i-1 >= 0 else False
        
        # Entry conditions: require squeeze + breakout in trend direction + volume
        long_entry = squeeze[i-1] and bull_bias and breakout_up and volume_confirmed if i-1 >= 0 else False
        short_entry = squeeze[i-1] and bear_bias and breakout_down and volume_confirmed if i-1 >= 0 else False
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price + (ATR_TARGET_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price - (ATR_TARGET_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals