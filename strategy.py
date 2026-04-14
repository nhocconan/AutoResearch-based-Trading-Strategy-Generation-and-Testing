#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 20-period EMA on 1d
    close_1d = df_1d['close'].values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 50-period EMA on 1d
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 14-period RSI on 1d
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Calculate 20-period standard deviation for Bollinger Bands on 1d
    sma_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma_20_1d + 2 * std_20_1d
    lower_bb_1d = sma_20_1d - 2 * std_20_1d
    
    # Calculate 4-period volume moving average (4h periods in a day)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(200, n):
        # Get aligned indicators
        ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)[i]
        ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)[i]
        rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)[i]
        upper_bb_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_bb_1d)[i]
        lower_bb_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_bb_1d)[i]
        vol_ma_4_val = vol_ma_4[i]  # already LTF
        
        # Check for NaN values
        if (np.isnan(ema_20_1d_aligned) or np.isnan(ema_50_1d_aligned) or 
            np.isnan(rsi_1d_aligned) or np.isnan(upper_bb_1d_aligned) or 
            np.isnan(lower_bb_1d_aligned) or np.isnan(vol_ma_4_val)):
            continue
        
        # Volume confirmation (> 1.3x average)
        volume_confirm = volume[i] > 1.3 * vol_ma_4_val
        
        # Trend filter: EMA20 > EMA50 for long, EMA20 < EMA50 for short
        trend_up = ema_20_1d_aligned > ema_50_1d_aligned
        trend_down = ema_20_1d_aligned < ema_50_1d_aligned
        
        if position == 0:  # No position - look for entries
            if volume_confirm:
                # Long: Pullback to support in uptrend (price touches lower BB + RSI oversold)
                if trend_up and close[i] <= lower_bb_1d_aligned and rsi_1d_aligned < 30:
                    position = 1
                    signals[i] = position_size
                # Short: Pullback to resistance in downtrend (price touches upper BB + RSI overbought)
                elif trend_down and close[i] >= upper_bb_1d_aligned and rsi_1d_aligned > 70:
                    position = -1
                    signals[i] = -position_size
        elif position == 1:  # Long position - exit when RSI overbought or trend breaks
            if rsi_1d_aligned > 70 or ema_20_1d_aligned < ema_50_1d_aligned:
                position = 0
                signals[i] = 0.0
        elif position == -1:  # Short position - exit when RSI oversold or trend breaks
            if rsi_1d_aligned < 30 or ema_20_1d_aligned > ema_50_1d_aligned:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_EMA20_50_RSI_BB_Pullback_v1"
timeframe = "4h"
leverage = 1.0