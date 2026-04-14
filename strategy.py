#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA Trend with RSI Momentum and Chop Filter
# Uses Kaufman's Adaptive Moving Average (KAMA) for trend direction
# RSI(14) for momentum confirmation
# Choppiness Index (14) to filter out ranging markets
# Designed for daily timeframe to capture major trends while avoiding whipsaws
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA (21) for trend direction
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate KAMA (10, 2, 30) for trend
    close_series = pd.Series(close)
    change = abs(close_series.diff(10))
    volatility = close_series.diff(1).abs().rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    sc = (er * (2/2 - 1/30) + 1/30) ** 2
    kama = [np.nan] * len(close)
    kama[9] = close.iloc[9] if len(close) > 9 else close[0]
    for i in range(10, len(close)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    kama = np.array(kama)
    
    # Calculate RSI (14)
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Calculate Choppiness Index (14)
    atr = []
    tr1 = high[1:] - low[1:]
    tr2 = abs(high[1:] - close[:-1])
    tr3 = abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_raw = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_raw * 14 / (max_high - min_low)) / np.log10(14)
    chop = np.nan_to_num(chop, nan=50.0)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 30  # for KAMA and RSI calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop[i]) or np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Trend filter: price above/below 1w EMA
        above_weekly_ema = price > ema_1w_aligned[i]
        
        # KAMA trend: price above/below KAMA
        above_kama = price > kama[i]
        
        # RSI momentum: not overbought/oversold
        rsi_momentum = (rsi[i] > 30) and (rsi[i] < 70)
        
        # Chop filter: trending market (chop < 61.8)
        trending = chop[i] < 61.8
        
        if position == 0:
            # Long: price above KAMA and weekly EMA, with momentum and trend
            if above_kama and above_weekly_ema and rsi_momentum and trending:
                position = 1
                signals[i] = position_size
            # Short: price below KAMA and weekly EMA, with momentum and trend
            elif (not above_kama) and (not above_weekly_ema) and rsi_momentum and trending:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price below KAMA or weekly EMA
            if (not above_kama) or (not above_weekly_ema):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price above KAMA or weekly EMA
            if above_kama or above_weekly_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_KAMA_RSI_Chop_Trend"
timeframe = "1d"
leverage = 1.0