#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour RSI with 1-day trend filter and volume confirmation
# Long when RSI(14) crosses above 30 (oversold bounce), 1d close > 1d EMA200 (uptrend), and volume > 1.5x 1h average volume
# Short when RSI(14) crosses below 70 (overbought rejection), 1d close < 1d EMA200 (downtrend), and volume > 1.5x 1h average volume
# Exit when RSI reverses (above 70 for long, below 30 for short) or trend changes
# Stoploss at 2.0 * ATR(14)
# Position size: 0.20 (20% of capital)
# Uses 1d EMA200 for trend filter and 1h volume average for confirmation
# Target: 75-150 total trades over 4 years (19-38/year)

name = "1h_rsi_1d_ema200_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1h RSI(14)
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 1h volume average for confirmation
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(200, n):
        # Skip if required data not available
        if (np.isnan(rsi[i]) or np.isnan(rsi[i-1]) or 
            np.isnan(ema200_1d_aligned[i]) or np.isnan(volume_ma[i]) or 
            np.isnan(atr[i])):
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
            # Exit: RSI > 70 (overbought) or trend changes
            elif rsi[i] >= 70 or close[i] < ema200_1d_aligned[i]:
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
            # Exit: RSI < 30 (oversold) or trend changes
            elif rsi[i] <= 30 or close[i] > ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with RSI reversal, trend alignment, and volume confirmation
            # Bullish: RSI crosses above 30 (oversold bounce)
            bullish_signal = rsi[i] > 30 and rsi[i-1] <= 30
            # Bearish: RSI crosses below 70 (overbought rejection)
            bearish_signal = rsi[i] < 70 and rsi[i-1] >= 70
            
            # Long: RSI bullish, 1d uptrend, volume spike
            if (bullish_signal and
                close[i] > ema200_1d_aligned[i] and
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: RSI bearish, 1d downtrend, volume spike
            elif (bearish_signal and
                  close[i] < ema200_1d_aligned[i] and
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
    
    return signals