#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_KAMA_Trend_RSI_MeanReversion"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once for trend and regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d KAMA for trend direction
    close_1d = df_1d['close'].values
    # KAMA parameters: ER period 10, fast EMA 2, slow EMA 30
    er_period = 10
    fast_ema = 2
    slow_ema = 30
    
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = np.power(er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1), 2)
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate 1d RSI for mean reversion
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 1d volatility (ATR) for regime filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close_1d[:-1])
    tr3 = np.abs(low[1:] - close_1d[:-1])
    tr = np.concatenate([[high[0] - low[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    
    # Align all 1d indicators to 4h timeframe
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    # Calculate 4h RSI for entry timing
    delta_4h = np.diff(close, prepend=close[0])
    gain_4h = np.where(delta_4h > 0, delta_4h, 0)
    loss_4h = np.where(delta_4h < 0, -delta_4h, 0)
    avg_gain_4h = pd.Series(gain_4h).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss_4h = pd.Series(loss_4h).ewm(alpha=1/14, adjust=False).mean().values
    rs_4h = np.where(avg_loss_4h != 0, avg_gain_4h / avg_loss_4h, 0)
    rsi_4h = 100 - (100 / (1 + rs_4h))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(rsi_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        kama_val = kama_1d_aligned[i]
        rsi_1d_val = rsi_1d_aligned[i]
        atr_val = atr_1d_aligned[i]
        rsi_4h_val = rsi_4h[i]
        
        # Volatility regime filter: only trade when volatility is elevated
        vol_threshold = np.nanmean(atr_1d_aligned[max(0, i-50):i+1]) * 1.2
        high_vol = atr_val > vol_threshold
        
        if position == 0:
            # Enter long: KAMA uptrend + RSI mean reversion from oversold
            if (close[i] > kama_val and 
                rsi_4h_val < 35 and 
                rsi_1d_val < 40 and 
                high_vol):
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA downtrend + RSI mean reversion from overbought
            elif (close[i] < kama_val and 
                  rsi_4h_val > 65 and 
                  rsi_1d_val > 60 and 
                  high_vol):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI mean reversion or trend change
            if (rsi_4h_val > 65 or close[i] < kama_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI mean reversion or trend change
            if (rsi_4h_val < 35 or close[i] > kama_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals