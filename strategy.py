#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d RSI(14) for mean reversion signals
    delta = pd.Series(df_1d['close'].values).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14_1d = (100 - (100 / (1 + rs))).values
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Calculate 1d Bollinger Bands (20,2) for volatility regime
    sma_20_1d = pd.Series(df_1d['close'].values).rolling(window=20, min_periods=20).mean().values
    std_20_1d = pd.Series(df_1d['close'].values).rolling(window=20, min_periods=20).std().values
    bb_upper_20_1d = sma_20_1d + 2 * std_20_1d
    bb_lower_20_1d = sma_20_1d - 2 * std_20_1d
    bb_upper_20_1d_aligned = align_htf_to_ltf(prices, df_1d, bb_upper_20_1d)
    bb_lower_20_1d_aligned = align_htf_to_ltf(prices, df_1d, bb_lower_20_1d)
    
    # Calculate 12h Donchian(10) for entry timing
    donchian_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    donchian_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(rsi_14_1d_aligned[i]) or 
            np.isnan(bb_upper_20_1d_aligned[i]) or np.isnan(bb_lower_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: Only trade when price is outside Bollinger Bands (high volatility expansion)
        price_bb_position = (close[i] - bb_lower_20_1d_aligned[i]) / (bb_upper_20_1d_aligned[i] - bb_lower_20_1d_aligned[i] + 1e-10)
        high_vol_regime = (price_bb_position < 0) or (price_bb_position > 1)
        
        # Long conditions:
        # 1. 1d EMA50 uptrend (price above EMA50)
        # 2. 1d RSI oversold (< 30) - mean reversion long
        # 3. High volatility regime (BB breakout)
        # 4. 12h price breaks above Donchian(10) high for entry
        if (close[i] > ema_50_1d_aligned[i] and
            rsi_14_1d_aligned[i] < 30 and
            high_vol_regime and
            close[i] > donchian_high_10[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 1d EMA50 downtrend (price below EMA50)
        # 2. 1d RSI overbought (> 70) - mean reversion short
        # 3. High volatility regime (BB breakout)
        # 4. 12h price breaks below Donchian(10) low for entry
        elif (close[i] < ema_50_1d_aligned[i] and
              rsi_14_1d_aligned[i] > 70 and
              high_vol_regime and
              close[i] < donchian_low_10[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_EMA50_RSI_BB_Donchian10_MeanRev_v1"
timeframe = "12h"
leverage = 1.0