#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA trend + 12h RSI mean reversion + volume spike.
# In bull market (KAMA rising), buy pullbacks when 12h RSI < 30 with volume spike.
# In bear market (KAMA falling), sell bounces when 12h RSI > 70 with volume spike.
# Works in both regimes by trading pullbacks in trend direction.
# Target: 25-35 trades/year with strict entry conditions.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h data (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # === 12h data (HTF for RSI and trend filter) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # === 4h KAMA (10,2,30) for trend direction ===
    def kama(close, er_length=10, fast=2, slow=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.abs(np.diff(close)).cumsum() - np.abs(np.diff(close, prepend=close[0])).cumsum()
        er = np.where(volatility != 0, change / volatility, 0)
        sc = np.power(er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1), 2)
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_4h = kama(close_4h, 10, 2, 30)
    kama_4h_series = pd.Series(kama_4h)
    kama_4h = kama_4h_series.ewm(span=1, adjust=False).values  # already smoothed
    
    # === 12h RSI(14) for mean reversion ===
    def rsi(close, length=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        gain_ma = pd.Series(gain).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values
        loss_ma = pd.Series(loss).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values
        rs = np.where(loss_ma != 0, gain_ma / loss_ma, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_12h = rsi(close_12h, 14)
    
    # === 4h volume spike confirmation ===
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_4h = volume_4h / vol_ma_20_4h
    
    # Align HTF indicators to 4h timeframe
    kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(kama_4h_aligned[i]) or np.isnan(rsi_12h_aligned[i]) or 
            np.isnan(vol_ratio_4h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        kama_val = kama_4h_aligned[i]
        rsi_val = rsi_12h_aligned[i]
        vol_ratio = vol_ratio_4h[i]
        
        # === STOPLOSS LOGIC ===
        if position == 1:  # Long position
            atr_4h = np.abs(high_4h - low_4h)
            atr_ma = pd.Series(atr_4h).rolling(window=14, min_periods=14).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_4h, atr_ma)
            atr_val = atr_aligned[i]
            if price < entry_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            atr_4h = np.abs(high_4h - low_4h)
            atr_ma = pd.Series(atr_4h).rolling(window=14, min_periods=14).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_4h, atr_ma)
            atr_val = atr_aligned[i]
            if price > entry_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when KAMA turns down or RSI > 70 (overbought)
            if kama_val < kama_4h_aligned[i-1] or rsi_val > 70:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when KAMA turns up or RSI < 30 (oversold)
            if kama_val > kama_4h_aligned[i-1] or rsi_val < 30:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require volume spike for confirmation
            if vol_ratio > 2.0:
                # Buy when KAMA trending up and RSI oversold (pullback in uptrend)
                if kama_val > kama_4h_aligned[i-1] and rsi_val < 30:
                    signals[i] = 0.30
                    position = 1
                    entry_price = price
                    continue
                # Sell when KAMA trending down and RSI overbought (bounce in downtrend)
                elif kama_val < kama_4h_aligned[i-1] and rsi_val > 70:
                    signals[i] = -0.30
                    position = -1
                    entry_price = price
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.30
        elif position == -1:
            signals[i] = -0.30
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_KAMA_RSI_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0