#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Hypothesis: 4h KAMA trend + 1d RSI mean reversion + volume spike
    # Long: 4h KAMA rising + 1d RSI < 30 (oversold) + 1d volume > 2x 20-period average
    # Short: 4h KAMA falling + 1d RSI > 70 (overbought) + 1d volume > 2x 20-period average
    # Exit: 4h KAMA reverses direction
    # Uses 4h primary for trend, 1d for mean reversion and volume confirmation
    # Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag
    # KAMA adapts to market noise, reducing whipsaws in choppy markets
    
    close = prices['close'].values
    
    # Get 4h data for primary timeframe (trend)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Get 1d data for RSI and volume (MTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.zeros(len(df_1d))
    
    # Calculate KAMA on 4h data (ER=10, fastest=2, slowest=30)
    # KAMA adapts its smoothing constant based on market efficiency
    close_4h_series = pd.Series(close_4h)
    change = np.abs(close_4h_series - close_4h_series.shift(10))
    volatility = close_4h_series.diff().abs().rolling(window=10, min_periods=1).sum()
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # smoothing constant
    kama = np.zeros_like(close_4h)
    kama[0] = close_4h[0]
    for i in range(1, len(close_4h)):
        kama[i] = kama[i-1] + sc[i] * (close_4h[i] - kama[i-1])
    
    # Calculate RSI on 1d data (14-period)
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 1d volume average (20-period)
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_4h, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi.values)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):  # start from 20 to have enough data for calculations
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_avg_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 2x 20-period 1d average
        curr_vol_1d = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        volume_confirmed = curr_vol_1d > 2.0 * vol_avg_20_aligned[i]
        
        # KAMA direction: rising or falling
        kama_rising = kama_aligned[i] > kama_aligned[i-1]
        kama_falling = kama_aligned[i] < kama_aligned[i-1]
        
        # RSI conditions: oversold/overbought
        rsi_oversold = rsi_aligned[i] < 30
        rsi_overbought = rsi_aligned[i] > 70
        
        # Entry conditions
        long_entry = kama_rising and rsi_oversold and volume_confirmed
        short_entry = kama_falling and rsi_overbought and volume_confirmed
        
        # Exit conditions: KAMA reverses direction
        exit_long = position == 1 and not kama_rising
        exit_short = position == -1 and not kama_falling
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_kama_rsi_volume_v1"
timeframe = "4h"
leverage = 1.0