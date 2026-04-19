#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camillo_Trend_Exhaustion_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for multi-timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    
    # 1d ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr_1d = np.maximum(high_1d - low_1d, np.absolute(high_1d - np.roll(close_1d, 1)), np.absolute(low_1d - np.roll(close_1d, 1)))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 1d EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 1d 10-period RSI for exhaustion signal
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/10, adjust=False, min_periods=10).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/10, adjust=False, min_periods=10).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_10 = 100 - (100 / (1 + rs))
    rsi_10_aligned = align_htf_to_ltf(prices, df_1d, rsi_10)
    
    # 4h 20-period RSI for entry confirmation
    delta_4h = pd.Series(close).diff()
    gain_4h = delta_4h.clip(lower=0)
    loss_4h = -delta_4h.clip(upper=0)
    avg_gain_4h = pd.Series(gain_4h).ewm(alpha=1/20, adjust=False, min_periods=20).mean().values
    avg_loss_4h = pd.Series(loss_4h).ewm(alpha=1/20, adjust=False, min_periods=20).mean().values
    rs_4h = avg_gain_4h / (avg_loss_4h + 1e-10)
    rsi_20_4h = 100 - (100 / (1 + rs_4h))
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 200, 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(atr_1d_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or \
           np.isnan(rsi_10_aligned[i]) or np.isnan(rsi_20_4h[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        rsi_4h = rsi_20_4h[i]
        rsi_1d = rsi_10_aligned[i]
        
        # Volume filter
        volume_ok = vol > 1.5 * vol_ma
        
        # Exhaustion: RSI > 70 in 1d indicates overextended long; RSI < 30 indicates oversold short
        long_exhaustion = rsi_1d > 70
        short_exhaustion = rsi_1d < 30
        
        # Entry conditions: look for reversal from exhaustion
        if position == 0:
            # Long: price above EMA200, RSI coming down from overbought, volume confirmation
            if price > ema200_1d_aligned[i] and rsi_4h < 50 and long_exhaustion and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: price below EMA200, RSI coming up from oversold, volume confirmation
            elif price < ema200_1d_aligned[i] and rsi_4h > 50 and short_exhaustion and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price drops below EMA200 or RSI shows new exhaustion
            if price < ema200_1d_aligned[i] or rsi_1d > 75:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price rises above EMA200 or RSI shows new exhaustion
            if price > ema200_1d_aligned[i] or rsi_1d < 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals