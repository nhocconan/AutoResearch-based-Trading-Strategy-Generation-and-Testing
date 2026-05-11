#!/usr/bin/env python3
name = "6h_4hTrend_6hMomentum_Confluence"
timeframe = "6h"
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
    
    # 4h trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    sma_20_4h = pd.Series(close_4h).rolling(window=20, min_periods=20).mean().values
    sma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, sma_20_4h)
    
    # 6h momentum: RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(sma_20_4h_aligned[i]) or np.isnan(rsi[i]) or np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        price_above_4h_sma = close[i] > sma_20_4h_aligned[i]
        price_below_4h_sma = close[i] < sma_20_4h_aligned[i]
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        if position == 0:
            # Long: Price above 4h SMA + RSI oversold + volume
            if price_above_4h_sma and rsi_oversold and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price below 4h SMA + RSI overbought + volume
            elif price_below_4h_sma and rsi_overbought and volume_ok[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit: Price crosses below 4h SMA OR RSI > 50
                if close[i] < sma_20_4h_aligned[i] or rsi[i] > 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit: Price crosses above 4h SMA OR RSI < 50
                if close[i] > sma_20_4h_aligned[i] or rsi[i] < 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals