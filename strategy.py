#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and ATR-based volatility filter
# Long when price breaks above 20-day high AND price > 1w EMA34 AND ATR(14) < ATR(50) (low volatility environment)
# Short when price breaks below 20-day low AND price < 1w EMA34 AND ATR(14) < ATR(50)
# Uses 1w EMA34 for higher-timeframe trend alignment to reduce whipsaws in ranging markets.
# ATR ratio filter ensures entries occur in low volatility periods, reducing false breakouts.
# Discrete sizing (0.25) minimizes fee churn. Target: 30-100 total trades over 4 years (7-25/year).

name = "1d_Donchian20_Breakout_1wEMA34_Trend_ATRFilter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1w HTF data for EMA calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # 1w EMA(34) on 1w close
    ema_1w_34 = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA to 1d timeframe
    ema_1w_34_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_34)
    
    # ATR calculation
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_14 / atr_50  # ATR ratio < 1 indicates low volatility
    
    # Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 50  # Need 50 for ATR(50) and Donchian(20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1w_34_aligned[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(atr_ratio[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # ATR filter: only trade in low volatility environments (ATR ratio < 1.0)
        vol_filter = atr_ratio[i] < 1.0
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above 20-day high, above 1w EMA34, low volatility
            if curr_high > highest_20[i-1] and curr_close > ema_1w_34_aligned[i] and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 20-day low, below 1w EMA34, low volatility
            elif curr_low < lowest_20[i-1] and curr_close < ema_1w_34_aligned[i] and vol_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on price below 20-day low or below 1w EMA34
            if curr_low < lowest_20[i-1] or curr_close < ema_1w_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on price above 20-day high or above 1w EMA34
            if curr_high > highest_20[i-1] or curr_close > ema_1w_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals