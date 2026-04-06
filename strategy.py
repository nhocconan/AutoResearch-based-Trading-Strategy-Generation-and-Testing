#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1w trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions; extreme readings (>80 or <20) signal reversals
# 1w EMA50 determines primary trend direction - only take trades in trend direction
# Volume confirmation: current volume > 1.5x 20-period average reduces false signals
# Target: 50-150 total trades over 4 years with controlled risk in both bull and bear markets
# Uses 6h timeframe with 1w trend filter to avoid counter-trend whipsaws

name = "6h_williamsr_1w_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Williams %R (14-period)
    def calculate_williams_r(high, low, close, period=14):
        highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
        wr = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
        return wr
    
    wr = calculate_williams_r(high, low, close, 14)
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1w data to 6h timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(wr[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(volume[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR approximation using price range
            if close[i] < entry_price - 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Williams %R returns from oversold or trend changes
            elif wr[i] > -20 or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR approximation
            if close[i] > entry_price + 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Williams %R returns from overbought or trend changes
            elif wr[i] < -80 or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend and volume filters
            # Only take longs in uptrend (price above EMA50) and shorts in downtrend
            if close[i] > ema50_1w_aligned[i]:  # Uptrend
                # Long when Williams %R is deeply oversold (< -80) with volume confirmation
                if wr[i] < -80 and volume[i] > 1.5 * vol_avg[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
            else:  # Downtrend
                # Short when Williams %R is deeply overbought (> -20) with volume confirmation
                if wr[i] > -20 and volume[i] > 1.5 * vol_avg[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals