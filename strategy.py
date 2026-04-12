#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly breakout from Bollinger Bands (20,2) with volume confirmation
# and daily RSI filter to avoid overtrading. Works in bull (breakouts continue)
# and bear (mean reversion at bands). Target: 10-20 trades/year.
name = "1d_1w_bollinger_breakout_volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Bollinger Bands
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly Bollinger Bands (20, 2)
    close_1w = pd.Series(df_1w['close'])
    bb_middle = close_1w.rolling(window=20, min_periods=20).mean().values
    bb_std = close_1w.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Align to daily
    bb_upper_aligned = align_htf_to_ltf(prices, df_1w, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1w, bb_lower)
    bb_middle_aligned = align_htf_to_ltf(prices, df_1w, bb_middle)
    
    # Weekly volume average (20-period)
    vol_1w = pd.Series(df_1w['volume']).rolling(window=20, min_periods=20).mean().values
    vol_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_1w)
    
    # Daily RSI (14) for filter
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or 
            np.isnan(bb_middle_aligned[i]) or np.isnan(vol_1w_aligned[i]) or 
            np.isnan(rsi_values[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current weekly volume > 20-period average
        volume_filter = volume[i] > vol_1w_aligned[i]
        
        # Long: price breaks above upper BB with volume and RSI < 70 (not overbought)
        long_signal = (close[i] > bb_upper_aligned[i] and volume_filter and rsi_values[i] < 70)
        
        # Short: price breaks below lower BB with volume and RSI > 30 (not oversold)
        short_signal = (close[i] < bb_lower_aligned[i] and volume_filter and rsi_values[i] > 30)
        
        # Exit: price returns to middle band
        exit_long = (position == 1 and close[i] < bb_middle_aligned[i])
        exit_short = (position == -1 and close[i] > bb_middle_aligned[i])
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals