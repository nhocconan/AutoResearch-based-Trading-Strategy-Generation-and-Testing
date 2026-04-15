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
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate daily Bollinger Bands (20, 2) for regime filter
    sma_20_1d = pd.Series(df_1d['close'].values).rolling(window=20, min_periods=20).mean().values
    std_20_1d = pd.Series(df_1d['close'].values).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma_20_1d + 2 * std_20_1d
    lower_bb_1d = sma_20_1d - 2 * std_20_1d
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb_1d)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb_1d)
    
    signals = np.zeros(n)
    position = 0  # track position to avoid whipsaw
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when daily ATR is elevated (> 0.3% of price)
        vol_filter = atr_14_1d_aligned[i] > 0.003 * close[i]
        
        # Regime filter: trade only when price is outside Bollinger Bands (trending regime)
        regime_filter = (close[i] > upper_bb_aligned[i]) or (close[i] < lower_bb_aligned[i])
        
        # Long conditions:
        # 1. Price above daily EMA34 (bullish bias)
        # 2. Price above upper Bollinger Band (strong uptrend)
        # 3. Volatility and regime filters
        if (close[i] > ema_34_1d_aligned[i] and
            close[i] > upper_bb_aligned[i] and
            vol_filter and
            regime_filter and
            position <= 0):  # avoid flipping
            signals[i] = 0.25
            position = 1
            
        # Short conditions:
        # 1. Price below daily EMA34 (bearish bias)
        # 2. Price below lower Bollinger Band (strong downtrend)
        # 3. Volatility and regime filters
        elif (close[i] < ema_34_1d_aligned[i] and
              close[i] < lower_bb_aligned[i] and
              vol_filter and
              regime_filter and
              position >= 0):  # avoid flipping
            signals[i] = -0.25
            position = -1
        else:
            # Hold position or flat
            if position == 1 and close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0  # exit long on trend reversal
                position = 0
            elif position == -1 and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0  # exit short on trend reversal
                position = 0
            else:
                signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
    
    return signals

name = "12h_EMA34_BBands_Trend_v1"
timeframe = "12h"
leverage = 1.0