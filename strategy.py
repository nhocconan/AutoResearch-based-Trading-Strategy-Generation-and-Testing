#!/usr/bin/env python3
"""
6h_momentum_12h1d_volume_v1
Hypothesis: On 6h timeframe, use RSI(14) momentum with volume confirmation, filtered by 12h and 1d trend alignment.
Enter long when RSI crosses above 50 with 12h and 1d uptrend and volume > 1.5x average.
Enter short when RSI crosses below 50 with 12h and 1d downtrend and volume > 1.5x average.
Exit when RSI crosses back across 50 or trend misalignment occurs.
Targets 12-37 trades/year to minimize fee dust while capturing sustained moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_momentum_12h1d_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI (14) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_series = pd.Series(gain)
    loss_series = pd.Series(loss)
    avg_gain = gain_series.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss_series.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 12h data for trend filter (calculate once before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate EMA20 and EMA50 on 12h close
    close_12h = df_12h['close'].values
    close_12h_s = pd.Series(close_12h)
    ema20_12h = close_12h_s.ewm(span=20, min_periods=20, adjust=False).mean().values
    ema50_12h = close_12h_s.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align to 6h timeframe (shifted by 1 12h bar to avoid look-ahead)
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Get 1d data for trend filter (calculate once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA20 and EMA50 on 1d close
    close_1d = df_1d['close'].values
    close_1d_s = pd.Series(close_1d)
    ema20_1d = close_1d_s.ewm(span=20, min_periods=20, adjust=False).mean().values
    ema50_1d = close_1d_s.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align to 6h timeframe (shifted by 1 1d bar to avoid look-ahead)
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):  # Start after RSI warmup
        # Skip if required data not available
        if (np.isnan(rsi_values[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0 or
            np.isnan(ema20_12h_aligned[i]) or np.isnan(ema50_12h_aligned[i]) or
            np.isnan(ema20_1d_aligned[i]) or np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: up if both EMAs >, down if both EMAs <
        trend_12h_up = ema20_12h_aligned[i] > ema50_12h_aligned[i]
        trend_12h_down = ema20_12h_aligned[i] < ema50_12h_aligned[i]
        trend_1d_up = ema20_1d_aligned[i] > ema50_1d_aligned[i]
        trend_1d_down = ema20_1d_aligned[i] < ema50_1d_aligned[i]
        
        trend_up = trend_12h_up and trend_1d_up
        trend_down = trend_12h_down and trend_1d_down
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit when RSI crosses below 50
            if rsi_values[i] < 50:
                exit_long = True
            # Exit on trend misalignment
            elif not trend_up:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit when RSI crosses above 50
            if rsi_values[i] > 50:
                exit_short = True
            # Exit on trend misalignment
            elif not trend_down:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: RSI crosses above 50, trend up, volume confirmation
            long_entry = (rsi_values[i] > 50) and (rsi_values[i-1] <= 50) and trend_up and vol_confirm
            
            # Short entry: RSI crosses below 50, trend down, volume confirmation
            short_entry = (rsi_values[i] < 50) and (rsi_values[i-1] >= 50) and trend_down and vol_confirm
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals