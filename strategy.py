#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d ATR for volatility filter ===
    df_1d = get_htf_data(prices, '1d')
    tr = np.maximum(df_1d['high'] - df_1d['low'], 
                    np.maximum(np.abs(df_1d['high'] - np.roll(df_1d['close'], 1)),
                               np.abs(df_1d['low'] - np.roll(df_1d['close'], 1))))
    tr[0] = df_1d['high'][0] - df_1d['low'][0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === 1d Bollinger Bands (20, 2.0) ===
    sma_20 = pd.Series(df_1d['close']).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(df_1d['close']).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2.0 * std_20
    lower_bb = sma_20 - 2.0 * std_20
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    # === 1d RSI(14) ===
    delta = pd.Series(df_1d['close']).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d.values)
    
    # === 6h Bollinger Band Width (20, 2.0) for regime detection ===
    sma_20_6h = pd.Series(close).rolling(window=20, min_periods=20).mean()
    std_20_6h = pd.Series(close).rolling(window=20, min_periods=20).std()
    upper_bb_6h = sma_20_6h + 2.0 * std_20_6h
    lower_bb_6h = sma_20_6h - 2.0 * std_20_6h
    bb_width_6h = (upper_bb_6h - lower_bb_6h) / sma_20_6h
    bb_width_6h = bb_width_6h.fillna(0).values
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 40
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or
            np.isnan(bb_width_6h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        atr = atr_1d_aligned[i]
        bbw = bb_width_6h[i]
        
        # Regime detection: trending when BB width is expanding
        # Use 10-period average of BB width for smoother regime detection
        if i >= 10:
            bbw_ma = np.mean(bb_width_6h[i-9:i+1])
        else:
            bbw_ma = bbw
        
        # Threshold for trending regime: BB width above median
        # We'll use a simple threshold that adapts to volatility
        trending_regime = bbw > np.percentile(bb_width_6h[max(0, i-100):i+1], 50) if i >= 100 else bbw > 0.01
        
        if position == 0:  # Flat - look for new entries
            # LONG: BB squeeze breakout in trending regime with RSI not overbought
            if (price > upper_bb_6h[i] and 
                trending_regime and 
                rsi_1d_aligned[i] < 70):
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: BB squeeze breakdown in trending regime with RSI not oversold
            elif (price < lower_bb_6h[i] and 
                  trending_regime and 
                  rsi_1d_aligned[i] > 30):
                signals[i] = -0.25
                position = -1
                continue
        
        elif position == 1:  # Long position - exit on mean reversion signal
            # Exit when price returns to middle of Bollinger Bands
            middle_bb_6h = (upper_bb_6h[i] + lower_bb_6h[i]) / 2
            if price < middle_bb_6h:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position - exit on mean reversion signal
            # Exit when price returns to middle of Bollinger Bands
            middle_bb_6h = (upper_bb_6h[i] + lower_bb_6h[i]) / 2
            if price > middle_bb_6h:
                signals[i] = 0.0
                position = 0
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_BBSqueeze_TrendingRegime_RSIFilter"
timeframe = "6h"
leverage = 1.0