#!/usr/bin/env python3
# Hypothesis: 12h KAMA trend with 1d RSI mean reversion and volume confirmation.
# Long when KAMA turns up, RSI < 30 (oversold), and volume > 1.5x average.
# Short when KAMA turns down, RSI > 70 (overbought), and volume > 1.5x average.
# Uses ATR trailing stop (2.5x) for risk control. Designed to catch trend reversals
# from extreme RSI levels with volume confirmation, effective in both bull and bear markets.
# Target: 15-30 trades/year (60-120 total over 4 years) on 12h timeframe.

name = "12h_KAMA_RSI_VolumeMeanRev_v1"
timeframe = "12h"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Get 12h data for KAMA trend
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate KAMA on 12h close
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_12h, n=10))
    volatility = np.sum(np.abs(np.diff(close_12h, n=1)), axis=1)
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.full_like(close_12h, np.nan)
    kama[9] = close_12h[9]  # Start after 10 periods
    for i in range(10, len(close_12h)):
        kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    kama_12h = kama
    
    # Align KAMA to 12h timeframe (wait for completed 12h bar)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    
    # Get 1d data for RSI
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate RSI(14) on 1d close
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    # Prepend NaN for first value
    rsi = np.concatenate([[np.nan], rsi])
    rsi_1d = rsi
    
    # Align RSI to 12h timeframe (wait for completed 1d bar)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate volume spike: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)
    lowest_since_entry = np.full(n, np.nan)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_12h_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: KAMA turning up, RSI < 30 (oversold), volume spike
            if (i > 0 and kama_12h_aligned[i] > kama_12h_aligned[i-1] and 
                rsi_1d_aligned[i] < 30 and volume_spike[i]):
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]
            # SHORT: KAMA turning down, RSI > 70 (overbought), volume spike
            elif (i > 0 and kama_12h_aligned[i] < kama_12h_aligned[i-1] and 
                  rsi_1d_aligned[i] > 70 and volume_spike[i]):
                signals[i] = -0.25
                position = -1
                lowest_since_entry[i] = low[i]
            else:
                signals[i] = 0.0
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
        elif position == 1:
            highest_since_entry[i] = max(highest_since_entry[i-1], high[i])
            # EXIT LONG: trailing stop (2.5x ATR) or RSI > 50 (exit overextended long)
            trailing_stop = close[i] < (highest_since_entry[i] - 2.5 * atr[i])
            rsi_exit = rsi_1d_aligned[i] > 50
            if trailing_stop or rsi_exit:
                signals[i] = 0.0
                position = 0
                highest_since_entry[i] = np.nan
            else:
                signals[i] = 0.25
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
        elif position == -1:
            lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
            # EXIT SHORT: trailing stop (2.5x ATR) or RSI < 50 (exit overextended short)
            trailing_stop = close[i] > (lowest_since_entry[i] + 2.5 * atr[i])
            rsi_exit = rsi_1d_aligned[i] < 50
            if trailing_stop or rsi_exit:
                signals[i] = 0.0
                position = 0
                lowest_since_entry[i] = np.nan
            else:
                signals[i] = -0.25
                if i > 0:
                    lowest_since_entry[i] = lowest_since_entry[i-1]
    
    return signals