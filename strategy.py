#!/usr/bin/env python3
"""
Hypothesis: 1-hour Stochastic RSI reversal combined with 4-hour EMA trend filter and volume confirmation.
Long when StochRSI < 10 (oversold), price > 4h EMA50 (uptrend), and volume > 1.5x average.
Short when StochRSI > 90 (overbought), price < 4h EMA50 (downtrend), and volume > 1.5x average.
Exit when StochRSI crosses back through 50 (mean reversion complete) or volume drops.
Designed for low trade frequency (~15-35/year) to capture mean reversion in ranging markets while avoiding false signals in strong trends.
Works in both bull and bear markets by requiring trend alignment (price vs 4h EMA50) and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4-hour data for EMA50 - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4-hour EMA50
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Stochastic RSI (14,14,3,3) on 1-hour
    rsi_period = 14
    stoch_period = 14
    k_period = 3
    d_period = 3
    
    # RSI calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Stochastic of RSI
    rsi_min = pd.Series(rsi_values).rolling(window=stoch_period, min_periods=stoch_period).min()
    rsi_max = pd.Series(rsi_values).rolling(window=stoch_period, min_periods=stoch_period).max()
    stoch_rsi = (rsi_values - rsi_min) / (rsi_max - rsi_min + 1e-10) * 100
    
    # %K and %D
    k = pd.Series(stoch_rsi).rolling(window=k_period, min_periods=k_period).mean()
    d = pd.Series(k).rolling(window=d_period, min_periods=d_period).mean()
    k_values = k.values
    d_values = d.values
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(k_values[i]) or np.isnan(d_values[i]) or 
            np.isnan(ema_4h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        k_val = k_values[i]
        d_val = d_values[i]
        ema_4h_val = ema_4h_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: StochRSI oversold (<10), price above 4h EMA50 (uptrend), volume confirmation
            if (k_val < 10 and d_val < 10 and k_values[i-1] >= d_values[i-1] and
                price > ema_4h_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.20
                position = 1
            # Short: StochRSI overbought (>90), price below 4h EMA50 (downtrend), volume confirmation
            elif (k_val > 90 and d_val > 90 and k_values[i-1] <= d_values[i-1] and
                  price < ema_4h_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: StochRSI crosses above 50 (mean reversion) OR volume drops
                if (k_val > 50 and k_values[i-1] <= 50) or vol_current < vol_ma_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: StochRSI crosses below 50 (mean reversion) OR volume drops
                if (k_val < 50 and k_values[i-1] >= 50) or vol_current < vol_ma_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_StochRSI_4hEMA50_Volume_MeanReversion"
timeframe = "1h"
leverage = 1.0