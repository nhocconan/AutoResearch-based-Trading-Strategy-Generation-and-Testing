#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d price closes above/below weekly 200 EMA + volume surge
# - Weekly EMA200 defines long-term trend (bullish if price above, bearish if below)
# - Daily close crosses the weekly EMA200 with volume > 2x 20-day average signals momentum shift
# - Works in bull markets (breaks above weekly EMA200 with volume) and bear (breaks below)
# - Weekly timeframe reduces noise; daily provides timely entry
# - Target: 10-25 trades/year (40-100 total over 4 years) to minimize fee drag
# - Position size: 0.25 to balance return and drawdown

name = "1d_1w_ema200_volume_breakout"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load weekly data ONCE before loop for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return signals
    
    # Pre-compute weekly EMA200
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Pre-compute daily volume SMA (20-period)
    volume_series = pd.Series(volume)
    volume_sma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    for i in range(50, n):  # Start after 50-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(ema200_1w_aligned[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 2x 20-day average
        vol_confirm = volume_current > 2.0 * volume_sma_20[i]
        
        # Price relative to weekly EMA200
        price_above_ema200 = price_close > ema200_1w_aligned[i]
        price_below_ema200 = price_close < ema200_1w_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Price crosses above weekly EMA200 with volume surge
        if price_above_ema200 and vol_confirm:
            enter_long = True
        
        # Short: Price crosses below weekly EMA200 with volume surge
        if price_below_ema200 and vol_confirm:
            enter_short = True
        
        # Exit conditions: price crosses back through weekly EMA200
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price crosses below weekly EMA200
            exit_long = price_below_ema200
        elif position == -1:
            # Exit short if price crosses above weekly EMA200
            exit_short = price_above_ema200
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals