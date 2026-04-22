#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA trend + 1d Bollinger Bands squeeze + volume confirmation.
# KAMA adapts to market noise, effective in both trending and ranging markets.
# Bollinger Bands squeeze (low volatility) precedes expansion; breakout direction follows KAMA trend.
# Volume confirms breakout strength. Designed for low-frequency, high-quality signals.
# Targets 20-30 trades/year to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for Bollinger Bands (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-period SMA and 2-standard deviation Bollinger Bands on 1d
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # Bollinger Band Width: (upper - lower) / middle
    bb_width = (upper_bb - lower_bb) / sma_20
    # Bollinger Band Width percentile over 50 days to identify squeeze (low volatility)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    # Squeeze condition: BB Width below 20th percentile (low volatility)
    squeeze = bb_width_percentile < 20
    
    # Align squeeze to 4h timeframe
    squeeze_aligned = align_htf_to_ltf(prices, df_1d, squeeze)
    
    # Calculate KAMA on 4h data
    close = prices['close'].values
    # Efficiency Ratio: abs(net change) / sum of absolute changes over 10 periods
    change = np.abs(np.diff(close, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # placeholder, will compute correctly below
    # Recalculate volatility properly: sum of absolute changes over 10 periods
    volatility = np.zeros_like(close)
    for i in range(10, len(close)):
        volatility[i] = np.sum(np.abs(np.diff(close[i-10:i+1])))
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants: fastest SC = 2/(2+1) = 0.67, slowest SC = 2/(30+1) = 0.0645
    sc = (er * (0.67 - 0.0645) + 0.0645) ** 2
    # Initialize KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period for indicators
        # Skip if data not ready
        if (np.isnan(kama[i]) or 
            np.isnan(squeeze_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        squeeze_val = squeeze_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long conditions: price above KAMA + volatility squeeze + volume spike
            if price > kama[i] and squeeze_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price below KAMA + volatility squeeze + volume spike
            elif price < kama[i] and squeeze_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: volatility expansion (end of squeeze) or opposite KAMA cross
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when volatility expands (squeeze ends) or price crosses below KAMA
                if not squeeze_val or price < kama[i]:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when volatility expands (squeeze ends) or price crosses above KAMA
                if not squeeze_val or price > kama[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_KAMA_BB_Squeeze_Volume"
timeframe = "4h"
leverage = 1.0