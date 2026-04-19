#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band squeeze breakout with 1d MACD trend filter and volume confirmation.
# Bollinger Band squeeze (low volatility) precedes explosive moves. Breakout in direction of 1d MACD trend.
# Uses Bollinger Bands width percentile to detect squeeze (<20th percentile = squeeze).
# MACD on 1d timeframe for trend direction: MACD line > signal line = bullish, < = bearish.
# Volume confirmation: current volume > 1.5x 20-period average.
# Target: 25-35 trades/year per symbol to stay within frequency limits.
name = "4h_BB_Squeeze_MACD_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for MACD calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate MACD (12,26,9)
    ema12 = pd.Series(close_1d).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema26 = pd.Series(close_1d).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema12 - ema26
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_hist = macd_line - signal_line  # Not used but calculated for completeness
    
    # Bollinger Bands (20,2) on 4h price data
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band width percentile (50-period lookback) to detect squeeze
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else np.nan, raw=False
    ).values
    
    # Align MACD and signal line to 4h timeframe
    macd_aligned = align_htf_to_ltf(prices, df_1d, macd_line)
    signal_aligned = align_htf_to_ltf(prices, df_1d, signal_line)
    
    # Get 4h average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 26, 20)  # Ensure BB width percentile (50), MACD (26), and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bb_width_percentile[i]) or np.isnan(macd_aligned[i]) or 
            np.isnan(signal_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        bb_width_pctl = bb_width_percentile[i]
        macd_val = macd_aligned[i]
        signal_val = signal_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Bollinger Band squeeze condition: width < 20th percentile
        is_squeeze = bb_width_pctl < 20.0
        
        # MACD trend filter: bullish if MACD > signal, bearish if MACD < signal
        is_bullish_trend = macd_val > signal_val
        is_bearish_trend = macd_val < signal_val
        
        # Volume confirmation
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Look for breakout in direction of 1d MACD trend during squeeze
            if is_squeeze and volume_confirmed:
                # Bullish breakout: price above upper BB and bullish 1d trend
                if price > bb_upper[i] and is_bullish_trend:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price below lower BB and bearish 1d trend
                elif price < bb_lower[i] and is_bearish_trend:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: price crosses below middle Bollinger Band or opposite band
            if price < sma20[i] or price < bb_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above middle Bollinger Band or opposite band
            if price > sma20[i] or price > bb_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals