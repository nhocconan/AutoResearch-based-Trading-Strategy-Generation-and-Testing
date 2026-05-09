#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Bollinger Band width regime + 1d RSI mean reversion + volume confirmation
# Bollinger Band width < 20th percentile indicates low volatility squeeze (mean reversion setup)
# RSI < 30 or > 70 on 1d timeframe signals overextended conditions
# Volume spike confirms institutional participation in the reversal
# Trades only when volatility is low and price is overextended, targeting mean reversion
# Works in both bull and bear markets as it fades extremes during low volatility periods
# Target: 15-30 trades/year (60-120 over 4 years) to avoid fee drag
name = "12h_BBWidth_RSI_MeanReversion_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for RSI and Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 20-period Bollinger Bands on 1d
    sma20 = pd.Series(df_1d['close']).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(df_1d['close']).rolling(window=20, min_periods=20).std().values
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    bb_width = (upper - lower) / sma20  # Normalized width
    
    # Percentile of bb_width over last 100 days (20th percentile threshold)
    bb_width_percentile = pd.Series(bb_width).rolling(window=100, min_periods=20).apply(
        lambda x: np.percentile(x, 20), raw=True
    ).values
    
    # 14-period RSI on 1d
    delta = pd.Series(df_1d['close']).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # 20-period volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 12h
    bb_width_percentile_12h = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    rsi_12h = align_htf_to_ltf(prices, df_1d, rsi_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for bb_width percentile
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bb_width_percentile_12h[i]) or np.isnan(rsi_12h[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility regime: bb_width below 20th percentile (low volatility squeeze)
        low_vol = bb_width_percentile_12h[i] < bb_width_percentile[i]
        
        # Volume condition: current volume > 1.8 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 1.8
        
        if position == 0:
            # Long: RSI < 30 (oversold) + low volatility + volume spike
            if rsi_12h[i] < 30 and low_vol and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) + low volatility + volume spike
            elif rsi_12h[i] > 70 and low_vol and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI crosses above 50 (neutral) or volatility expands
            if rsi_12h[i] > 50 or not low_vol:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI crosses below 50 (neutral) or volatility expands
            if rsi_12h[i] < 50 or not low_vol:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals