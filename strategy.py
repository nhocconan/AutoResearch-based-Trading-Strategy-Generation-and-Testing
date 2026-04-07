#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour MACD histogram reversal with 4-hour RSI filter and 1-day volume confirmation
# Long when MACD histogram crosses above zero + 4h RSI > 50 + 1d volume > 1.2x 20-period average
# Short when MACD histogram crosses below zero + 4h RSI < 50 + 1d volume > 1.2x 20-period average
# Exit when MACD histogram crosses back to opposite side (zero-line crossover)
# Stoploss at 2.0 * ATR(14)
# Position size: 0.20 (20% of capital)
# Uses multi-timeframe confirmation to filter false signals and reduce trade frequency
# Target: 60-150 total trades over 4 years (15-37/year)

name = "1h_macd_hist_4h_rsi_1d_vol_v1"
timeframe = "1h"
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
    
    # 4-hour data for RSI filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    # 1-day data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 4-hour RSI (14-period)
    close_4h = df_4h['close'].values
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_4h = 100 - (100 / (1 + rs))
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Calculate 1-day volume average (20-period)
    volume_1d = df_1d['volume'].values
    volume_1d_s = pd.Series(volume_1d)
    volume_ma = volume_1d_s.rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma)
    
    # Calculate MACD (12,26,9) on 1-hour data
    ema_fast = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema_slow = pd.Series(close).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_hist = macd_line - signal_line
    
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
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(rsi_4h_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(macd_hist[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: MACD histogram crosses below zero
            elif macd_hist[i] < 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: MACD histogram crosses above zero
            elif macd_hist[i] > 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: MACD histogram zero-cross with 4h RSI filter and 1d volume confirmation
            # Volume filter: volume > 1.2x 20-period daily average
            volume_filter = volume[i] > 1.2 * volume_ma_aligned[i]
            # RSI filter: RSI > 50 for long, RSI < 50 for short
            rsi_filter_long = rsi_4h_aligned[i] > 50
            rsi_filter_short = rsi_4h_aligned[i] < 50
            
            # Long: MACD histogram crosses above zero + volume filter + RSI > 50
            if macd_hist[i] > 0 and macd_hist[i-1] <= 0 and volume_filter and rsi_filter_long:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: MACD histogram crosses below zero + volume filter + RSI < 50
            elif macd_hist[i] < 0 and macd_hist[i-1] >= 0 and volume_filter and rsi_filter_short:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
    
    return signals