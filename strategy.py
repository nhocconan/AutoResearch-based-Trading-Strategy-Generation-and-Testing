#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Bollinger Bands with RSI confirmation and volume filter.
# Enters long when price closes below weekly BB lower band with RSI < 30 and volume > 1.5x average.
# Enters short when price closes above weekly BB upper band with RSI > 70 and volume > 1.5x average.
# Exits when price returns to weekly BB middle band or RSI reverts to neutral (40-60).
# Designed for ~15-25 trades/year by requiring extreme conditions (BB bands + RSI extremes).
# Works in bull/bear: buys oversold dips, sells overbought rallies.
# Uses weekly timeframe for trend context to avoid counter-trend trades in strong trends.
# Volume filter ensures breakouts have conviction.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Bollinger Bands and RSI
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly Bollinger Bands (20, 2)
    bb_length = 20
    bb_mult = 2.0
    
    # Basis (SMA)
    basis_1w = pd.Series(close_1w).rolling(window=bb_length, min_periods=bb_length).mean().values
    # Deviation
    dev_1w = bb_mult * pd.Series(close_1w).rolling(window=bb_length, min_periods=bb_length).std().values
    # Upper and lower bands
    upper_1w = basis_1w + dev_1w
    lower_1w = basis_1w - dev_1w
    
    # Calculate weekly RSI (14)
    rsi_length = 14
    delta = pd.Series(close_1w).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=rsi_length, min_periods=rsi_length).mean()
    avg_loss = loss.rolling(window=rsi_length, min_periods=rsi_length).mean()
    rs = avg_gain / avg_loss
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w = rsi_1w.fillna(50).values  # Fill NaN with neutral 50
    
    # Align weekly indicators to daily
    basis_aligned = align_htf_to_ltf(prices, df_1w, basis_1w)
    upper_aligned = align_htf_to_ltf(prices, df_1w, upper_1w)
    lower_aligned = align_htf_to_ltf(prices, df_1w, lower_1w)
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Volume filter: volume > 1.5 x 20-period average (daily)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need weekly BB (20), RSI (14), volume MA (20)
    start_idx = max(20, 14, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(basis_aligned[i]) or np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or
            np.isnan(rsi_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 1.5 * vol_avg
        
        # RSI extremes
        rsi_oversold = rsi_aligned[i] < 30
        rsi_overbought = rsi_aligned[i] > 70
        rsi_neutral = (rsi_aligned[i] >= 40) & (rsi_aligned[i] <= 60)
        
        if position == 0:
            # Long: price closes below lower BB with RSI oversold and volume
            if price < lower_aligned[i] and rsi_oversold and vol_filter:
                signals[i] = size
                position = 1
            # Short: price closes above upper BB with RSI overbought and volume
            elif price > upper_aligned[i] and rsi_overbought and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to basis or RSI becomes neutral
            if price > basis_aligned[i] or rsi_neutral:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to basis or RSI becomes neutral
            if price < basis_aligned[i] or rsi_neutral:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WeeklyBB_RSI_Volume"
timeframe = "1d"
leverage = 1.0