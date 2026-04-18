#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for indicators
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily ATR(14) with proper min_periods
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align daily ATR14 to 12h timeframe
    atr_14d_aligned = align_htf_to_ltf(prices, df_1d, atr_14d)
    
    # Calculate daily ATR(40) for volatility regime filter
    tr1_40 = high_1d - low_1d
    tr2_40 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_40 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1_40[0] = high_1d[0] - low_1d[0]
    tr2_40[0] = np.abs(high_1d[0] - close_1d[0])
    tr3_40[0] = np.abs(low_1d[0] - close_1d[0])
    tr_40 = np.maximum(tr1_40, np.maximum(tr2_40, tr3_40))
    atr_40d = pd.Series(tr_40).ewm(alpha=1/40, adjust=False, min_periods=40).mean().values
    
    # Align daily ATR40 to 12h timeframe
    atr_40d_aligned = align_htf_to_ltf(prices, df_1d, atr_40d)
    
    # Calculate 12-period ATR on 12h timeframe
    tr_12h_1 = high - low
    tr_12h_2 = np.abs(high - np.roll(close, 1))
    tr_12h_3 = np.abs(low - np.roll(close, 1))
    tr_12h_1[0] = high[0] - low[0]
    tr_12h_2[0] = np.abs(high[0] - close[0])
    tr_12h_3[0] = np.abs(low[0] - close[0])
    tr_12h = np.maximum(tr_12h_1, np.maximum(tr_12h_2, tr_12h_3))
    atr_12h = pd.Series(tr_12h).ewm(alpha=1/12, adjust=False, min_periods=12).mean().values
    
    # Calculate 12h volume moving average (24-period)
    vol_ma_12h = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma_12h[i] = np.mean(volume[i-24:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 24)  # need daily ATR40 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr_14d_aligned[i]) or np.isnan(atr_40d_aligned[i]) or 
            np.isnan(vol_ma_12h[i]) or np.isnan(atr_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: ATR14 > ATR40 indicates high volatility regime
        high_vol_regime = atr_14d_aligned[i] > atr_40d_aligned[i]
        
        # Volume confirmation: current volume > 1.8 * 24-period average
        vol_confirmed = volume[i] > 1.8 * vol_ma_12h[i]
        
        # Price momentum: close > open indicates bullish momentum
        bullish_momentum = close[i] > open_price[i]
        bearish_momentum = close[i] < open_price[i]
        
        if position == 0:
            # Long entry: bullish momentum + high volatility + volume confirmation
            if bullish_momentum and high_vol_regime and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish momentum + high volatility + volume confirmation
            elif bearish_momentum and high_vol_regime and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: momentum reversal or volatility drop
            if not bullish_momentum or not high_vol_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: momentum reversal or volatility drop
            if not bearish_momentum or not high_vol_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_VolRegime_Momentum_Volume"
timeframe = "12h"
leverage = 1.0