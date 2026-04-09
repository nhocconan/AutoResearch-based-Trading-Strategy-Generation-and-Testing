#!/usr/bin/env python3
# 6h_ema200_rsi_pullback_1d_v1
# Hypothesis: 6h EMA200 trend + RSI(14) pullback with 1d EMA50 filter.
# Works in bull/bear: 1d EMA50 defines higher-timeframe trend; 6h EMA200 defines intermediate trend;
# RSI pullbacks (long when RSI<40 in uptrend, short when RSI>60 in downtrend) capture mean reversion within trend.
# Volume confirmation ensures institutional participation. Target: 12-37 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ema200_rsi_pullback_1d_v1"
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
    
    # 1d HTF data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 6h EMA200 for intermediate trend
    ema200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # RSI(14) on 6h
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(ema200[i]) or np.isnan(rsi[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below EMA200 OR RSI > 70 (overbought)
            if close[i] < ema200[i] or rsi[i] > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above EMA200 OR RSI < 30 (oversold)
            if close[i] > ema200[i] or rsi[i] < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation
            volume_confirmed = volume[i] > 1.5 * volume_ma[i]
            
            if volume_confirmed:
                # Long: price above both EMAs AND RSI pullback from oversold (<40)
                if close[i] > ema200[i] and close[i] > ema50_1d_aligned[i] and rsi[i] < 40:
                    position = 1
                    signals[i] = 0.25
                # Short: price below both EMAs AND RSI pullback from overbought (>60)
                elif close[i] < ema200[i] and close[i] < ema50_1d_aligned[i] and rsi[i] > 60:
                    position = -1
                    signals[i] = -0.25
    
    return signals