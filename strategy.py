#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(2) mean reversion with 4h trend filter and 1d volatility regime filter
# RSI(2) < 10 for long, > 90 for short captures extreme short-term reversals
# 4h EMA50 trend filter ensures we trade with higher timeframe momentum
# 1d ATR ratio (ATR5/ATR20) < 0.8 identifies low volatility regimes where mean reversion works best
# Session filter (08-20 UTC) reduces noise during off-hours
# Target: 80-120 total trades over 4 years (20-30/year) on 1h timeframe

name = "1h_RSI2_MeanRev_4hEMA50_1dATRRegime"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 1 or len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d ATR5 and ATR20 for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    atr_5 = pd.Series(tr).rolling(window=5, min_periods=5).mean().values
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr_5 / atr_20  # Ratio of short-term to long-term volatility
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate RSI(2) on 1h timeframe
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_ma = pd.Series(gain).ewm(span=2, adjust=False, min_periods=2).mean().values
    loss_ma = pd.Series(loss).ewm(span=2, adjust=False, min_periods=2).mean().values
    rs = gain_ma / (loss_ma + 1e-10)
    rsi_2 = 100 - (100 / (1 + rs))
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 2)  # 4h EMA50 warmup, RSI(2) warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi_2[i]) or 
            np.isnan(atr_ratio_aligned[i])):
            signals[i] = 0.0
            continue
        
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        curr_rsi = rsi_2[i]
        curr_ema_4h = ema_50_4h_aligned[i]
        curr_atr_ratio = atr_ratio_aligned[i]
        curr_close = close[i]
        
        # Volatility regime filter: only trade in low volatility (mean reversion favorable)
        vol_regime = curr_atr_ratio < 0.8
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: RSI(2) > 50 (mean reversion complete) OR price breaks below 4h EMA50
            if curr_rsi > 50 or curr_close < curr_ema_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI(2) < 50 (mean reversion complete) OR price breaks above 4h EMA50
            if curr_rsi < 50 or curr_close > curr_ema_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Long entry: RSI(2) < 10 (extreme oversold) + price above 4h EMA50 + low volatility regime
            if (curr_rsi < 10 and 
                curr_close > curr_ema_4h and 
                vol_regime):
                signals[i] = 0.20
                position = 1
            # Short entry: RSI(2) > 90 (extreme overbought) + price below 4h EMA50 + low volatility regime
            elif (curr_rsi > 90 and 
                  curr_close < curr_ema_4h and 
                  vol_regime):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
    
    return signals