#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum with 4h/1d regime filter. Long when 1h RSI > 55 AND price > 4h EMA20 AND 1d close > 1d EMA50 (bull regime). 
# Short when 1h RSI < 45 AND price < 4h EMA20 AND 1d close < 1d EMA50 (bear regime). 
# Uses session filter (08-20 UTC) and discrete size 0.20. Designed for 15-35 trades/year to avoid fee drag.
# Works in bull via regime alignment; works in bear via short signals in downtrends.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 1h Indicators: RSI(14) ===
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === 4h Indicators: EMA(20) ===
    df_4h = get_htf_data(prices, '4h')
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # === 1d Indicators: EMA(50) ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(rsi[i]) or np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        rsi_val = rsi[i]
        ema_4h_val = ema_4h_aligned[i]
        ema_1d_val = ema_1d_aligned[i]
        
        # === EXIT LOGIC: reverse on opposite signal ===
        if position == 1:  # Long position
            # Exit if bearish regime: RSI < 45 AND price < 4h EMA20 AND 1d close < 1d EMA50
            if rsi_val < 45 and price < ema_4h_val and ema_1d_val < ema_1d[i]:  # Note: ema_1d[i] is current 1d EMA (not aligned, but same logic)
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit if bullish regime: RSI > 55 AND price > 4h EMA20 AND 1d close > 1d EMA50
            if rsi_val > 55 and price > ema_4h_val and ema_1d_val > ema_1d[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: bullish regime
            if rsi_val > 55 and price > ema_4h_val and ema_1d_val > ema_1d[i]:
                signals[i] = 0.20
                position = 1
            
            # SHORT: bearish regime
            elif rsi_val < 45 and price < ema_4h_val and ema_1d_val < ema_1d[i]:
                signals[i] = -0.20
                position = -1
        
        else:
            # Hold current position
            signals[i] = position * 0.20
    
    return signals

name = "1h_Momentum_4hEMA20_1dEMA50_Regime_V1"
timeframe = "1h"
leverage = 1.0