#!/usr/bin/env python3
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
    
    # === 1d data (primary timeframe) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d KAMA(14) for trend direction
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    dir = np.abs(np.diff(close_1d, prepend=close_1d[0]) - np.diff(close_1d, prepend=close_1d[0], n=2))
    er = np.where(change != 0, dir / change, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama = kama  # already array
    
    # 1d RSI(14) for momentum
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === 1w data (HTF for regime) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # 1w ATR(14) for volatility regime
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Align HTF data to 1d timeframe ===
    kama_1d = kama  # already on 1d
    rsi_1d = rsi    # already on 1d
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # === 1d ATR for volatility filter ===
    tr1d = np.abs(high_1d - low_1d)
    tr2d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3d = np.abs(low_1d - np.roll(close_1d, 1))
    tr2d[0] = np.inf
    tr3d[0] = np.inf
    trd = np.maximum(tr1d, np.maximum(tr2d, tr3d))
    atr_1d = pd.Series(trd).rolling(window=14, min_periods=14).mean().values
    
    # === Volume spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma_20
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_1d[i]) or np.isnan(rsi_1d[i]) or 
            np.isnan(atr_1w_aligned[i]) or np.isnan(atr_1d[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        kama_val = kama_1d[i]
        rsi_val = rsi_1d[i]
        atr_1w_val = atr_1w_aligned[i]
        atr_1d_val = atr_1d[i]
        vol_ratio_val = vol_ratio[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below KAMA OR RSI becomes overbought
            if (price < kama_val) or (rsi_val > 70):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above KAMA OR RSI becomes oversold
            if (price > kama_val) or (rsi_val < 30):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price above KAMA (uptrend) AND RSI not overbought AND 
            # volatility contraction (weekly ATR declining) AND volume expansion
            if (price > kama_val) and (rsi_val < 60) and \
               (atr_1w_val < atr_1w_aligned[i-1]) and (vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price below KAMA (downtrend) AND RSI not oversold AND 
            # volatility contraction AND volume expansion
            elif (price < kama_val) and (rsi_val > 40) and \
                 (atr_1w_val < atr_1w_aligned[i-1]) and (vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_KAMA_RSI_Vol_VolContraction"
timeframe = "1d"
leverage = 1.0