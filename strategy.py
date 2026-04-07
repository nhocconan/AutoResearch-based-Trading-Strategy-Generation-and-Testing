#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h RSI(14) mean reversion with 1d trend filter and volume confirmation
# Long when RSI < 30 + 1d close > 1d SMA(50) + volume > 1.5x 1d average volume
# Short when RSI > 70 + 1d close < 1d SMA(50) + volume > 1.5x 1d average volume
# Exit when RSI crosses back above 50 (long) or below 50 (short)
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses 1-day SMA for trend filter and 1-day volume for confirmation
# Target: 50-150 total trades over 4 years (12-37/year)

name = "6h_rsi_meanrev_1d_trend_vol_v1"
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
    
    # 1-day data for trend filter and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1-day close and volume
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1-day SMA(50) for trend filter
    close_1d_s = pd.Series(close_1d)
    sma_50_1d = close_1d_s.rolling(window=50, min_periods=50).mean().values
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    # 1-day average volume (50-period)
    volume_1d_s = pd.Series(volume_1d)
    volume_avg_1d = volume_1d_s.rolling(window=50, min_periods=50).mean().values
    volume_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_avg_1d)
    
    # RSI(14) on 6h data
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_s = pd.Series(gain)
    loss_s = pd.Series(loss)
    
    avg_gain = gain_s.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = loss_s.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
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
    
    for i in range(14, n):
        # Skip if required data not available
        if (np.isnan(rsi[i]) or np.isnan(sma_50_1d_aligned[i]) or 
            np.isnan(volume_avg_1d_aligned[i]) or np.isnan(atr[i])):
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
            # Exit: RSI crosses back above 50
            elif rsi[i] > 50:
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
            # Exit: RSI crosses back below 50
            elif rsi[i] < 50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: RSI extreme with trend filter and volume confirmation
            # Trend filter: 1d close > 1d SMA(50) for long, < for short
            trend_filter_long = close_1d[i] > sma_50_1d_aligned[i]
            trend_filter_short = close_1d[i] < sma_50_1d_aligned[i]
            # Volume filter: volume > 1.5x 1d average volume
            volume_filter = volume[i] > 1.5 * volume_avg_1d_aligned[i]
            
            # Long: RSI < 30 + trend filter long + volume filter
            if rsi[i] < 30 and trend_filter_long and volume_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: RSI > 70 + trend filter short + volume filter
            elif rsi[i] > 70 and trend_filter_short and volume_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals