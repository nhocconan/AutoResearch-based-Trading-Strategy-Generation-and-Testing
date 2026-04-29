#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter and volume confirmation
# Long when Williams %R(14) crosses above -80 (oversold) AND price > 1d EMA(50) AND volume > 1.5x 20-period average
# Short when Williams %R(14) crosses below -20 (overbought) AND price < 1d EMA(50) AND volume > 1.5x 20-period average
# Uses discrete position sizing (0.25) to minimize fee drag. Works in both bull and bear by following HTF trend.
# Timeframe: 6h (primary), HTF: 1d for trend filter and Williams %R calculation.
# Added ATR-based volatility filter to avoid ranging markets.

name = "6h_WilliamsR_MeanReversion_1dEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA(50)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R(14) on 1d data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14) * -100
    # Replace division by zero with -50 (neutral)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate ATR for volatility filter (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma_20 = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 50)  # warmup for indicators
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_volume = volume[i]
        curr_ema = ema_50_1d_aligned[i]
        curr_williams = williams_r_aligned[i]
        curr_atr_ma = atr_ma_20[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_confirm = volume[i] > 1.5 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Volatility filter: avoid extremely low volatility ranging markets
        vol_filter = curr_atr_ma > 0 and atr[i] > 0.5 * curr_atr_ma
        
        # Handle exits
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Williams %R crosses below -50 (momentum weakening)
            # 2. Price < 1d EMA(50) (trend change)
            if (curr_williams < -50 or curr_close < curr_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Williams %R crosses above -50 (momentum weakening)
            # 2. Price > 1d EMA(50) (trend change)
            if (curr_williams > -50 or curr_close > curr_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Williams %R crosses above -80 (from oversold) AND price > 1d EMA(50) AND volume confirm AND vol filter
            if i > start_idx:
                prev_williams = williams_r_aligned[i-1]
                williams_cross_up = prev_williams <= -80 and curr_williams > -80
                if (williams_cross_up and 
                    curr_close > curr_ema and 
                    vol_confirm and 
                    vol_filter):
                    signals[i] = 0.25
                    position = 1
            # Short entry: Williams %R crosses below -20 (from overbought) AND price < 1d EMA(50) AND volume confirm AND vol filter
            if i > start_idx:
                prev_williams = williams_r_aligned[i-1]
                williams_cross_down = prev_williams >= -20 and curr_williams < -20
                if (williams_cross_down and 
                    curr_close < curr_ema and 
                    vol_confirm and 
                    vol_filter):
                    signals[i] = -0.25
                    position = -1
            # Default flat
            if signals[i] == 0.0:
                signals[i] = 0.0
    
    return signals