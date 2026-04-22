#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Bollinger Band width contraction with 1d RSI mean reversion and volume spike confirmation.
# Bollinger Band width contraction indicates low volatility and potential for explosive moves.
# Combine with 1d RSI extremes (<30 or >70) for mean reversion signals.
# Volume spike (>2x 20-period average) confirms institutional participation.
# This strategy aims for low trade frequency (~15-25/year) by requiring multiple confluence factors.
# Works in both bull and bear markets by capturing volatility expansions from contraction phases.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Bollinger Bands and RSI calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Bollinger Bands (20-period, 2 standard deviations)
    bb_middle = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Calculate Bollinger Band width percentile (252-period lookback for annual context)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=252, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Calculate 1d RSI (14-period)
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # Fill NaN with neutral 50
    
    # Align 1d indicators to 12h timeframe (waits for 1d bar to close)
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(bb_width_percentile_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        bb_width_percentile_val = bb_width_percentile_aligned[i]
        rsi_val = rsi_aligned[i]
        
        # Bollinger Band width contraction: width < 20th percentile (low volatility)
        bb_width_contraction = bb_width_percentile_val < 20
        
        # Volume filter: current volume > 2.0 * 20-period average (strict filter)
        vol_spike = vol > 2.0 * vol_ma
        
        # RSI extremes for mean reversion
        rsi_oversold = rsi_val < 30
        rsi_overbought = rsi_val > 70
        
        if position == 0:
            # Long conditions: BB width contraction + RSI oversold + volume spike
            if bb_width_contraction and rsi_oversold and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: BB width contraction + RSI overbought + volume spike
            elif bb_width_contraction and rsi_overbought and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: volatility expansion or RSI returns to neutral
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when volatility expands (BB width > 50th percentile) or RSI > 50
                if bb_width_percentile_val > 50 or rsi_val > 50:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when volatility expands (BB width > 50th percentile) or RSI < 50
                if bb_width_percentile_val > 50 or rsi_val < 50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_BBWidth_RSI_MeanReversion_Volume"
timeframe = "12h"
leverage = 1.0