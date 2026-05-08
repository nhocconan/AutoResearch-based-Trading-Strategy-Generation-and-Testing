#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_BollingerBreakout_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w Bollinger Bands once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma_20 = pd.Series(close_1w).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_20 = pd.Series(close_1w).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma_20 + (bb_std * std_20)
    lower_bb = sma_20 - (bb_std * std_20)
    
    # Align Bollinger Bands to 6h timeframe (wait for weekly bar close)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1w, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1w, lower_bb)
    
    # Weekly trend: EMA50
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for EMA50 and BB
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above upper BB + uptrend + volume spike
            long_cond = (close[i] > upper_bb_aligned[i]) and \
                        (close[i] > ema_50_1w_aligned[i]) and \
                        volume_spike[i]
            # Short: break below lower BB + downtrend + volume spike
            short_cond = (close[i] < lower_bb_aligned[i]) and \
                         (close[i] < ema_50_1w_aligned[i]) and \
                         volume_spike[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: close below middle BB (mean reversion)
            middle_bb_aligned = sma_20[-1] if len(sma_20) > 0 else np.nan  # placeholder, will be replaced
            # Recalculate middle BB aligned for current index
            sma_20_aligned = align_htf_to_ltf(prices, df_1w, sma_20)
            if i < len(sma_20_aligned) and not np.isnan(sma_20_aligned[i]):
                if close[i] < sma_20_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close above middle BB (mean reversion)
            sma_20_aligned = align_htf_to_ltf(prices, df_1w, sma_20)
            if i < len(sma_20_aligned) and not np.isnan(sma_20_aligned[i]):
                if close[i] > sma_20_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25
    
    return signals