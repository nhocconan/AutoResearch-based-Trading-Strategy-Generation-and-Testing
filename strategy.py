#!/usr/bin/env python3
"""
1d_1w_kelly_sizing_v1
Strategy: Daily Kelly sizing based on weekly trend and daily mean reversion
Timeframe: 1d
Leverage: 1.0
Hypothesis: Combines weekly trend filter with daily mean reversion signals using Kelly criterion for position sizing. Uses weekly EMA for trend and daily RSI for mean reversion. Kelly sizing reduces position size during uncertain times and increases during high-probability setups, improving risk-adjusted returns. Designed to work in both bull and bear markets by adapting position size to signal confidence.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kelly_sizing_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Daily RSI(14) for mean reversion
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Daily Bollinger Bands (20, 2) for volatility and mean reversion
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean()
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std()
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    upper_bb_values = upper_bb.values
    lower_bb_values = lower_bb.values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(rsi_values[i]) or 
            np.isnan(upper_bb_values[i]) or np.isnan(lower_bb_values[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        
        # Weekly trend filter
        uptrend_1w = price_close > ema_20_1w_aligned[i]
        downtrend_1w = price_close < ema_20_1w_aligned[i]
        
        # Daily mean reversion signals
        rsi_oversold = rsi_values[i] < 30
        rsi_overbought = rsi_values[i] > 70
        price_at_lower_bb = price_low <= lower_bb_values[i]
        price_at_upper_bb = price_high >= upper_bb_values[i]
        
        # Long setup: weekly uptrend + daily oversold at BB lower band
        long_setup = uptrend_1w and rsi_oversold and price_at_lower_bb
        # Short setup: weekly downtrend + daily overbought at BB upper band
        short_setup = downtrend_1w and rsi_overbought and price_at_upper_bb
        
        # Exit conditions
        exit_long = position == 1 and (rsi_values[i] > 50 or price_close >= sma_20.iloc[i])
        exit_short = position == -1 and (rsi_values[i] < 50 or price_close <= sma_20.iloc[i])
        
        # Kelly sizing approximation based on signal strength
        # Base size: 0.25, scaled by RSI extremity and trend alignment
        if long_setup or short_setup:
            # Calculate RSI extremity (0 to 1)
            if long_setup:
                rsi_extremity = (30 - rsi_values[i]) / 30  # 0 at RSI=30, 1 at RSI=0
            else:  # short_setup
                rsi_extremity = (rsi_values[i] - 70) / 30  # 0 at RSI=70, 1 at RSI=100
            rsi_extremity = max(0, min(1, rsi_extremity))
            
            # Kelly fraction approximation: increase size with signal strength
            # Base Kelly fraction for 60% win rate, 1:1 payoff is ~20%
            # Scale from 0.15 to 0.30 based on signal strength
            kelly_size = 0.15 + (0.15 * rsi_extremity)
            
            if long_setup:
                position = 1
                signals[i] = kelly_size
            elif short_setup:
                position = -1
                signals[i] = -kelly_size
        else:
            # Hold current position or flat
            if position == 1:
                signals[i] = 0.25  # Base position size
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals