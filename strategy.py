#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R reversal with 1d EMA50 trend filter and ATR-based volatility filter.
Long when Williams %R crosses above -80 from below AND close > 1d EMA50 AND ATR(14) < 0.5 * ATR(50).
Short when Williams %R crosses below -20 from above AND close < 1d EMA50 AND ATR(14) < 0.5 * ATR(50).
Exit when Williams %R reverts to -50 or ATR-based stoploss hits.
Williams %R identifies overbought/oversold conditions; EMA50 filters trend direction; ATR ratio ensures low volatility environment for mean reversion.
Target: 50-150 total trades over 4 years = 12-37/year. Uses discrete position sizing (0.25) to minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data for Williams %R and EMA50 - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R(14) on 1d data: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align 1d Williams %R to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate EMA50 on 1d data
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate ATR(14) and ATR(50) on 6h data for volatility filter
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]  # first bar
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(atr14[i]) or np.isnan(atr50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        price = close[i]
        williams_r_val = williams_r_aligned[i]
        atr14_val = atr14[i]
        atr50_val = atr50[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 from below AND close > 1d EMA50 AND low volatility
            if (williams_r_val > -80 and 
                williams_r_aligned[i-1] <= -80 and  # crossed above -80
                close[i] > ema50_1d_aligned[i] and 
                atr14_val < 0.5 * atr50_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Williams %R crosses below -20 from above AND close < 1d EMA50 AND low volatility
            elif (williams_r_val < -20 and 
                  williams_r_aligned[i-1] >= -20 and  # crossed below -20
                  close[i] < ema50_1d_aligned[i] and 
                  atr14_val < 0.5 * atr50_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R reverts to -50 or ATR stoploss
                if williams_r_val >= -50:
                    exit_signal = True
                elif price < entry_price - 2.0 * atr14_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R reverts to -50 or ATR stoploss
                if williams_r_val <= -50:
                    exit_signal = True
                elif price > entry_price + 2.0 * atr14_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_Reversal_1dEMA50_VolatilityFilter"
timeframe = "6h"
leverage = 1.0