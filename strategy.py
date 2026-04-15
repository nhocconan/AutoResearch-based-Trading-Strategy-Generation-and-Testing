#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum with 4h/1d regime filter
# Long when 1h RSI(14) crosses above 30 (oversold bounce) AND price > 4h EMA50 AND price > 1d EMA200 (bull regime)
# Short when 1h RSI(14) crosses below 70 (overbought rejection) AND price < 4h EMA50 AND price < 1d EMA200 (bear regime)
# Uses 4h/1d EMAs for regime alignment to avoid counter-trend trades
# RSI crossovers provide timely entries with built-in mean reversion edge
# Designed for low trade frequency (15-35/year) to minimize fee drag while capturing swings
# Works in bull markets via regime filter and in bear markets via short signals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # === 4h Indicators: EMA50 for trend filter ===
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # === 1d Indicators: EMA200 for regime filter ===
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # === 1h Indicators: RSI(14) for momentum ===
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or
            np.isnan(rsi_values[i]) or np.isnan(rsi_values[i-1])):
            signals[i] = 0.0
            continue
        
        # RSI crossover conditions
        rsi_cross_up = (rsi_values[i-1] <= 30) and (rsi_values[i] > 30)
        rsi_cross_down = (rsi_values[i-1] >= 70) and (rsi_values[i] < 70)
        
        # === LONG CONDITIONS ===
        # 1. RSI crosses above 30 (oversold bounce)
        # 2. Price above 4h EMA50 (4h uptrend)
        # 3. Price above 1d EMA200 (bull regime)
        if rsi_cross_up and (close[i] > ema_50_4h_aligned[i]) and (close[i] > ema_200_1d_aligned[i]):
            signals[i] = 0.20
        
        # === SHORT CONDITIONS ===
        # 1. RSI crosses below 70 (overbought rejection)
        # 2. Price below 4h EMA50 (4h downtrend)
        # 3. Price below 1d EMA200 (bear regime)
        elif rsi_cross_down and (close[i] < ema_50_4h_aligned[i]) and (close[i] < ema_200_1d_aligned[i]):
            signals[i] = -0.20
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1h_RSI_Crossover_4hEMA50_1dEMA200_RegimeFilter_v1"
timeframe = "1h"
leverage = 1.0