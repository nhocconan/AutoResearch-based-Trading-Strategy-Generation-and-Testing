#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams %R mean reversion with daily trend filter and volume confirmation
# Long when Williams %R < -80 (oversold) + daily close > daily SMA50 (uptrend) + volume > 1.5x 20-period average
# Short when Williams %R > -20 (overbought) + daily close < daily SMA50 (downtrend) + volume > 1.5x 20-period average
# Exit when Williams %R crosses back above -50 (for long) or below -50 (for short)
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Target: 75-150 total trades over 4 years (19-38/year)

name = "6h_williamsr_meanrev_1d_trend_vol_v1"
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
    
    # 1-day data for trend filter (SMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day SMA50
    close_1d = df_1d['close'].values
    sma50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    
    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Volume filter: 20-period average
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(williams_r[i]) or np.isnan(sma50_1d_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Williams %R crosses above -50
            elif williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Williams %R crosses below -50
            elif williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Williams %R extreme + daily trend + volume confirmation
            # Volume filter: volume > 1.5x 20-period average
            volume_filter = volume[i] > 1.5 * volume_ma[i]
            # Trend filter: daily close vs SMA50
            trend_filter_long = close[i] > sma50_1d_aligned[i]
            trend_filter_short = close[i] < sma50_1d_aligned[i]
            
            # Long: Williams %R < -80 (oversold) + uptrend + volume
            if williams_r[i] < -80 and trend_filter_long and volume_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: Williams %R > -20 (overbought) + downtrend + volume
            elif williams_r[i] > -20 and trend_filter_short and volume_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals