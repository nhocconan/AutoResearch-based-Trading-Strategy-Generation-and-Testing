#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy combining daily volatility breakout with weekly RSI trend filter.
# Uses volatility-adjusted breakout levels based on prior day's range and ATR, providing adaptive support/resistance.
# Long when price breaks above volatility-adjusted resistance with 1w RSI > 50 (uptrend) and volume confirmation.
# Short when price breaks below volatility-adjusted support with 1w RSI < 50 (downtrend) and volume confirmation.
# Exit when price returns to prior day's close or RSI crosses 50 in opposite direction.
# Designed to work in both bull and bear markets by adapting to volatility and using RSI for trend confirmation.
# Target: 20-25 trades/year per symbol (80-100 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for volatility-adjusted levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR on 1d
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr_period = 14
    atr_1d = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Volatility-adjusted support/resistance: prior day's high/low ± 0.5 * ATR
    var_resistance = np.roll(high_1d, 1) + 0.5 * np.roll(atr_1d, 1)
    var_support = np.roll(low_1d, 1) - 0.5 * np.roll(atr_1d, 1)
    var_resistance[0] = np.nan
    var_support[0] = np.nan
    
    # Prior 1d close for exit condition
    prior_close_1d = np.roll(close_1d, 1)
    prior_close_1d[0] = np.nan
    
    # Load 1w data ONCE for RSI trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate RSI(14) on 1w
    delta = np.diff(close_1w, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / avg_loss
    rs = np.where(avg_loss == 0, 100, rs)
    rsi_1w = 100 - (100 / (1 + rs))
    
    # Align indicators to lower timeframe
    var_resistance_aligned = align_htf_to_ltf(prices, df_1d, var_resistance)
    var_support_aligned = align_htf_to_ltf(prices, df_1d, var_support)
    prior_close_1d_aligned = align_htf_to_ltf(prices, df_1d, prior_close_1d)
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 14)  # Need VAR and RSI
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(var_resistance_aligned[i]) or 
            np.isnan(var_support_aligned[i]) or
            np.isnan(prior_close_1d_aligned[i]) or
            np.isnan(rsi_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: RSI > 50 for uptrend, < 50 for downtrend
        uptrend = rsi_1w_aligned[i] > 50
        downtrend = rsi_1w_aligned[i] < 50
        
        if position == 0:
            # Look for volatility-adjusted breakouts
            # Long: price breaks above VAR resistance AND uptrend
            if (close[i] > var_resistance_aligned[i] and 
                uptrend and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below VAR support AND downtrend
            elif (close[i] < var_support_aligned[i] and 
                  downtrend and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to prior 1d close or RSI crosses below 50
            if (close[i] <= prior_close_1d_aligned[i] or 
                rsi_1w_aligned[i] <= 50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to prior 1d close or RSI crosses above 50
            if (close[i] >= prior_close_1d_aligned[i] or 
                rsi_1w_aligned[i] >= 50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_VolatilityAdjustedBreakout_1wRSI_v1"
timeframe = "4h"
leverage = 1.0