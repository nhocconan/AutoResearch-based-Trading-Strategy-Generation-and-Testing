#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Stochastic RSI Pullback with 1d Trend Filter
# Hypothesis: Stochastic RSI pullbacks in direction of 1d EMA(50) trend capture mean reversion within trends.
# Uses 1d EMA for trend filter (works in bull/bear) and Stochastic RSI for precise entries.
# Target: 20-35 trades/year (80-140 total over 4 years) to minimize fee drag.

name = "4h_stochrsi_pullback_1d_ema_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d close
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    
    # Align 1d EMA to 4h
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Stochastic RSI(14,14,3,3) on 4h
    # First calculate RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Then Stochastic of RSI
    rsi_high = pd.Series(rsi).rolling(window=14, min_periods=14).max().values
    rsi_low = pd.Series(rsi).rolling(window=14, min_periods=14).min().values
    stoch_rsi = (rsi - rsi_low) / (rsi_high - rsi_low) * 100
    stoch_rsi = np.where((rsi_high - rsi_low) == 0, 50, stoch_rsi)  # avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if required data not available
        if np.isnan(ema_50_aligned[i]) or np.isnan(stoch_rsi[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Stochastic RSI reaches overbought or trend changes
            if stoch_rsi[i] >= 80 or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: Stochastic RSI reaches oversold or trend changes
            if stoch_rsi[i] <= 20 or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Stochastic RSI pullback in direction of 1d trend
            if close[i] > ema_50_aligned[i]:  # Uptrend
                if stoch_rsi[i] <= 20:  # Pullback to buy (oversold)
                    position = 1
                    signals[i] = 0.25
            else:  # Downtrend
                if stoch_rsi[i] >= 80:  # Pullback to sell (overbought)
                    position = -1
                    signals[i] = -0.25
    
    return signals