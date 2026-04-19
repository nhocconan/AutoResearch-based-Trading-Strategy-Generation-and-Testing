#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band squeeze with 1d RSI trend filter and volume confirmation
# Bollinger Band squeeze (low volatility) precedes breakouts. Direction determined by 1d RSI:
# RSI > 50 = bullish bias, RSI < 50 = bearish bias. Volume confirmation avoids false breakouts.
# Works in both bull and bear markets as squeeze-breakout captures volatility expansion.
# Target: 20-30 trades/year per symbol with disciplined entries
name = "4h_BollingerSqueeze_1dRSI_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily RSI14 for trend bias
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate RSI on daily data
    delta = np.diff(df_1d['close'])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14_1d = 100 - (100 / (1 + rs))
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Bollinger Bands (20, 2) on 4h data
    bb_length = 20
    bb_mult = 2.0
    bb_basis = pd.Series(close).rolling(window=bb_length, min_periods=bb_length).mean().values
    bb_dev = bb_mult * pd.Series(close).rolling(window=bb_length, min_periods=bb_length).std().values
    bb_upper = bb_basis + bb_dev
    bb_lower = bb_basis - bb_dev
    
    # Bollinger Band width (normalized)
    bb_width = (bb_upper - bb_lower) / bb_basis
    # Squeeze condition: BB width below 20-period average (low volatility)
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze_condition = bb_width < bb_width_ma
    
    # Volume confirmation: volume > 1.3 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(rsi_14_1d_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: BB squeeze + price breaks above upper band + RSI > 50 (bullish) + volume confirmation
            if (squeeze_condition[i] and 
                close[i] > bb_upper[i] and 
                rsi_14_1d_aligned[i] > 50 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: BB squeeze + price breaks below lower band + RSI < 50 (bearish) + volume confirmation
            elif (squeeze_condition[i] and 
                  close[i] < bb_lower[i] and 
                  rsi_14_1d_aligned[i] < 50 and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price returns to middle band or volatility expands (width > MA)
            if (close[i] < bb_basis[i]) or (bb_width[i] > bb_width_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price returns to middle band or volatility expands (width > MA)
            if (close[i] > bb_basis[i]) or (bb_width[i] > bb_width_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals