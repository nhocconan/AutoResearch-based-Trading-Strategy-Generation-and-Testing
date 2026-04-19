#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_RSI_Momentum_Filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d and 1w data for multi-timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # 1d RSI for momentum filter
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 1w ATR for volatility filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr_1w = np.maximum(high_1w - low_1w, np.absolute(high_1w - np.roll(close_1w, 1)), np.absolute(low_1w - np.roll(close_1w, 1)))
    tr_1w[0] = high_1w[0] - low_1w[0]
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 6h momentum: price change over 2 periods (12h)
    mom_6h = (close - np.roll(close, 2)) / np.roll(close, 2)
    mom_6h[:2] = 0
    
    # 6h volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(rsi_1d_aligned[i]) or np.isnan(atr_1w_aligned[i]) or \
           np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(mom_6h[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        mom = mom_6h[i]
        
        # Volume filter
        volume_ok = vol > 1.5 * vol_ma
        
        # Momentum filter
        mom_ok = abs(mom) > 0.01  # 1% momentum threshold
        
        # Trend bias: long bias if price > EMA50, short bias if price < EMA50
        long_bias = price > ema50_1w_aligned[i]
        short_bias = price < ema50_1w_aligned[i]
        
        # RSI filter: avoid extreme overbought/oversold
        rsi = rsi_1d_aligned[i]
        rsi_ok_long = rsi < 70  # Not overbought
        rsi_ok_short = rsi > 30  # Not oversold
        
        if position == 0:
            # Long: bullish momentum + above EMA50 + RSI not overbought + volume
            if mom > 0 and mom_ok and long_bias and rsi_ok_long and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: bearish momentum + below EMA50 + RSI not oversold + volume
            elif mom < 0 and mom_ok and short_bias and rsi_ok_short and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: momentum turns bearish or RSI becomes overbought
            if mom < -0.005 or rsi > 75:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: momentum turns bullish or RSI becomes oversold
            if mom > 0.005 or rsi < 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals