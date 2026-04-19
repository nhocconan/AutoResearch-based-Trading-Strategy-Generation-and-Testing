#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h RSI divergence + 1d MACD trend filter with volume confirmation
# Uses RSI divergence to catch reversals in overbought/oversold conditions
# Only trades when 1d MACD confirms trend direction and volume spikes
# Works in bull markets via bullish divergence + MACD > 0
# Works in bear markets via bearish divergence + MACD < 0
# Target: 15-37 trades/year to avoid fee drag
name = "1h_RSIDivergence_MACDTrend_1d_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for multi-timeframe analysis (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d MACD for trend filter
    close_1d = df_1d['close'].values
    ema12_1d = pd.Series(close_1d).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema26_1d = pd.Series(close_1d).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema12_1d - ema26_1d
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_hist = macd_line - signal_line
    macd_line_aligned = align_htf_to_ltf(prices, df_1d, macd_line)
    signal_line_aligned = align_htf_to_ltf(prices, df_1d, signal_line)
    macd_hist_aligned = align_htf_to_ltf(prices, df_1d, macd_hist)
    
    # RSI for divergence detection (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: current volume > 2x average volume (24-period)
    if len(volume) >= 24:
        avg_volume = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    else:
        avg_volume = np.full_like(volume, volume[0])
    volume_filter = volume > 2 * avg_volume
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(rsi[i]) or np.isnan(macd_line_aligned[i]) or np.isnan(signal_line_aligned[i]) or np.isnan(volume_filter[i]):
            signals[i] = 0.0
            continue
        
        # RSI divergence detection (look back 5 periods for swing points)
        if i >= 5:
            # Bullish divergence: price makes lower low, RSI makes higher low
            price_lower_low = close[i] < close[i-5] and close[i] == np.min(close[i-5:i+1])
            rsi_higher_low = rsi[i] > rsi[i-5] and rsi[i] == np.min(rsi[i-5:i+1])
            bullish_divergence = price_lower_low and rsi_higher_low
            
            # Bearish divergence: price makes higher high, RSI makes lower high
            price_higher_high = close[i] > close[i-5] and close[i] == np.max(close[i-5:i+1])
            rsi_lower_high = rsi[i] < rsi[i-5] and rsi[i] == np.max(rsi[i-5:i+1])
            bearish_divergence = price_higher_high and rsi_lower_high
        else:
            bullish_divergence = False
            bearish_divergence = False
        
        if position == 0:
            # Long: Bullish divergence + MACD line above signal + volume filter
            if bullish_divergence and macd_line_aligned[i] > signal_line_aligned[i] and volume_filter[i]:
                signals[i] = 0.20
                position = 1
            # Short: Bearish divergence + MACD line below signal + volume filter
            elif bearish_divergence and macd_line_aligned[i] < signal_line_aligned[i] and volume_filter[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: Bearish divergence or MACD cross below signal
            if bearish_divergence or macd_line_aligned[i] < signal_line_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: Bullish divergence or MACD cross above signal
            if bullish_divergence or macd_line_aligned[i] > signal_line_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals