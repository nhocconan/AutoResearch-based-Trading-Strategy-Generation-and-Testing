#!/usr/bin/env python3
# 4h_triple_confluence_breakout_v1
# Hypothesis: Combines Donchian breakout (trend), volume confirmation (conviction), and 12h RSI filter (momentum) to capture strong moves while avoiding false breakouts.
# Works in bull/bear: Donchian catches breakouts, volume filters low-conviction moves, RSI avoids overextended entries.
# Target: 20-40 trades/year with strict 3-condition entry.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_triple_confluence_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period) on 4h
    donch_period = 20
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_high = high_series.rolling(window=donch_period, min_periods=donch_period).max().values
    donch_low = low_series.rolling(window=donch_period, min_periods=donch_period).min().values
    
    # Volume confirmation: 1.5x 20-period average
    vol_ma_period = 20
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=vol_ma_period, min_periods=vol_ma_period).mean().values
    vol_surge = volume > 1.5 * vol_ma
    
    # 12h RSI filter (momentum/overbought/oversold)
    df_12h = get_htf_data(prices, '12h')
    rsi_period = 14
    close_12h = pd.Series(df_12h['close'].values)
    delta = close_12h.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss
    rsi_12h = 100 - (100 / (1 + rs))
    rsi_12h_values = rsi_12h.values
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h_values)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(donch_period, vol_ma_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(rsi_12h_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price below Donchian low or RSI overbought (>70)
            if close[i] < donch_low[i] or rsi_12h_aligned[i] > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above Donchian high or RSI oversold (<30)
            if close[i] > donch_high[i] or rsi_12h_aligned[i] < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price above Donchian high + volume surge + RSI not overbought (<70)
            if (close[i] > donch_high[i] and vol_surge[i] and rsi_12h_aligned[i] < 70):
                position = 1
                signals[i] = 0.25
            # Short entry: Price below Donchian low + volume surge + RSI not oversold (>30)
            elif (close[i] < donch_low[i] and vol_surge[i] and rsi_12h_aligned[i] > 30):
                position = -1
                signals[i] = -0.25
    
    return signals