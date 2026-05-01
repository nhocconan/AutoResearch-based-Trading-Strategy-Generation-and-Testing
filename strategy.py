#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA direction filter with weekly EMA50 trend and volume spike confirmation.
# Long when KAMA(10,2,30) turns upward AND price > weekly EMA50 AND volume > 2.0x 20-day volume median.
# Short when KAMA turns downward AND price < weekly EMA50 AND volume > 2.0x 20-day volume median.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.5*ATR.
# KAMA adapts to market noise, weekly EMA50 provides strong trend filter, volume spike confirms momentum.
# Target: 10-20 trades/year on 1d timeframe (40-80 total over 4 years) to minimize fee drag.
# This combination has shown strong test performance in DB for SOL with proper filtering and should work on BTC/ETH.

name = "1d_KAMA_Direction_1wEMA50_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate KAMA(10,2,30) - Kaufman Adaptive Moving Average
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # 10-period sum of absolute changes
    # Fix volatility calculation - need rolling sum
    volatility = pd.Series(np.abs(np.diff(close, n=1))).rolling(window=10, min_periods=10).sum().values
    volatility = np.concatenate([np.full(9, np.nan), volatility])  # align lengths
    er = np.where(volatility != 0, change / volatility, 0)
    # SC = [ER * (fastest - slowest) + slowest]^2
    fastest = 2.0 / (2 + 1)   # EMA(2)
    slowest = 2.0 / (30 + 1)  # EMA(30)
    sc = (er * (fastest - slowest) + slowest) ** 2
    # Initialize KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # start at first available point
    for i in range(10, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate KAMA direction (1 = upward, -1 = downward, 0 = flat)
    kama_diff = np.diff(kama, n=1)
    kama_direction = np.where(kama_diff > 0, 1, np.where(kama_diff < 0, -1, 0))
    kama_direction = np.concatenate([[0], kama_direction])  # align length
    
    # Calculate weekly EMA50 trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 20-day volume median for confirmation
    vol_median_20d = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, KAMA, EMA, and volume
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(kama_direction[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_median_20d[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 2.0x 20-day volume median
        if vol_median_20d[i] <= 0 or np.isnan(vol_median_20d[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20d[i] * 2.0)
        
        # Trend filter: price vs weekly EMA50
        uptrend = curr_close > ema_50_1w_aligned[i]
        downtrend = curr_close < ema_50_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: KAMA turning up AND uptrend AND volume confirmation
            if (kama_direction[i] == 1 and 
                uptrend and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: KAMA turning down AND downtrend AND volume confirmation
            elif (kama_direction[i] == -1 and 
                  downtrend and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: KAMA turns down OR trend turns down
            elif (kama_direction[i] == -1) or (not uptrend):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: KAMA turns up OR trend turns up
            elif (kama_direction[i] == 1) or (not downtrend):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals