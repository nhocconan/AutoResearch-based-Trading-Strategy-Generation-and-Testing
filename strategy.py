#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band breakout with volume confirmation and volume-weighted RSI filter.
# Works in bull/bear: BB breakout captures momentum, volume confirms institutional interest,
# VW-RSI filters overextended moves. Low trade frequency via strict BB(20,2.0) and volume spike.
name = "4h_Bollinger_Band_Breakout_Volume_VWRSI"
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
    
    # Bollinger Bands (20, 2.0)
    close_series = pd.Series(close)
    bb_mid = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    
    # Volume-weighted RSI (14) - less prone to whipsaws
    # Calculate typical price and volume-weighted gains/losses
    typical_price = (high + low + close) / 3.0
    price_change = np.diff(typical_price, prepend=typical_price[0])
    
    gains = np.where(price_change > 0, price_change, 0.0)
    losses = np.where(price_change < 0, -price_change, 0.0)
    
    # Volume-weighted smoothing
    vol_series = pd.Series(volume)
    vol_weighted_gains = pd.Series(gains) * vol_series
    vol_weighted_losses = pd.Series(losses) * vol_series
    
    avg_gain = vol_weighted_gains.rolling(window=14, min_periods=14).mean().values / \
               vol_series.rolling(window=14, min_periods=14).mean().values
    avg_loss = vol_weighted_losses.rolling(window=14, min_periods=14).mean().values / \
               vol_series.rolling(window=14, min_periods=14).mean().values
    
    # Avoid division by zero
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    vw_rsi = 100 - (100 / (1 + rs))
    
    # Volume spike detection (20-period average)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for BB to stabilize
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(vw_rsi[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma20[i]  # Strong volume spike
        # VW-RSI filter: avoid overbought/oversold extremes
        rsi_not_extreme = (vw_rsi[i] > 20) and (vw_rsi[i] < 80)
        
        if position == 0:
            # Long: close breaks above upper BB + volume spike + RSI not extreme
            if (close[i] > bb_upper[i] and vol_ok and rsi_not_extreme):
                signals[i] = 0.25
                position = 1
            # Short: close breaks below lower BB + volume spike + RSI not extreme
            elif (close[i] < bb_lower[i] and vol_ok and rsi_not_extreme):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close crosses below middle BB or volume dries up
            if (close[i] < bb_mid[i] or volume[i] < 0.5 * vol_ma20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close crosses above middle BB or volume dries up
            if (close[i] > bb_mid[i] or volume[i] < 0.5 * vol_ma20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals