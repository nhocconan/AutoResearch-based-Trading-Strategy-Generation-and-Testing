#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Williams %R with 4h trend and 1d volatility filter
# Long when Williams %R crosses above -80 in uptrend (close > 4h EMA50) and low volatility (ATR ratio < 1.2)
# Short when Williams %R crosses below -20 in downtrend (close < 4h EMA50) and low volatility
# Exit when Williams %R crosses opposite threshold or volatility spikes
# Designed for low trade frequency (~20-40/year) to minimize fee drain. Works in range markets via mean reversion and in trends via trend filter.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 50-period EMA on 4h close for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Load 1d data for volatility filter (ATR)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ATR on 1d
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 50-period ATR average for volatility regime
    atr_ma_50_1d = pd.Series(atr_14_1d).rolling(window=50, min_periods=50).mean().values
    atr_ratio_1d = atr_14_1d / atr_ma_50_1d
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Calculate Williams %R (14-period) on 1h closes
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    close_1h = prices['close'].values
    
    highest_high = pd.Series(high_1h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1h) / (highest_high - lowest_low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(williams_r[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        wr = williams_r[i]
        ema_val = ema_50_4h_aligned[i]
        vol_ratio = atr_ratio_aligned[i]
        
        # Williams %R signals
        wr_cross_up = wr > -80 and williams_r[i-1] <= -80
        wr_cross_down = wr < -20 and williams_r[i-1] >= -20
        
        # Volatility filter: low volatility regime (ATR ratio < 1.2)
        low_vol = vol_ratio < 1.2
        
        if position == 0:
            # Long conditions: Williams %R crosses above -80 + uptrend + low volatility
            if wr_cross_up and price > ema_val and low_vol:
                signals[i] = 0.20
                position = 1
            # Short conditions: Williams %R crosses below -20 + downtrend + low volatility
            elif wr_cross_down and price < ema_val and low_vol:
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit conditions: Williams %R crosses opposite threshold or volatility spikes
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when Williams %R crosses below -20 or volatility spikes
                if wr_cross_down or vol_ratio >= 1.5:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when Williams %R crosses above -80 or volatility spikes
                if wr_cross_up or vol_ratio >= 1.5:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_WilliamsR_4hEMA50_1dATR_Volatility"
timeframe = "1h"
leverage = 1.0