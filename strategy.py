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
    
    # Get 1d data for regime filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 14-period ATR on 1d for volatility regime
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 50-period SMA on 1d for trend filter
    sma_50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    
    # Align 1d indicators to 12h
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    # Calculate 12-period RSI on 12h for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=12, min_periods=12).mean().values
    avg_loss = pd.Series(loss).rolling(window=12, min_periods=12).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 12h volume spike (volume > 1.8x 24-period average)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14, 12, 24) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr_1d_aligned[i]) or 
            np.isnan(sma_50_1d_aligned[i]) or
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: volatility expansion (ATR > 1.2x its 50-period MA)
        atr_ma = pd.Series(atr_1d_aligned).rolling(window=50, min_periods=50).mean().values
        vol_expansion = atr_1d_aligned[i] > (1.2 * atr_ma[i])
        
        # Trend filter: price above/below 1d SMA50
        uptrend = close[i] > sma_50_1d_aligned[i]
        downtrend = close[i] < sma_50_1d_aligned[i]
        
        # Momentum filter: RSI in favorable territory
        rsi_bullish = rsi[i] > 50
        rsi_bearish = rsi[i] < 50
        
        # Volume confirmation
        vol_confirmed = volume_spike[i]
        
        if position == 0:
            # Long: uptrend + bullish momentum + vol expansion + volume spike
            if uptrend and rsi_bullish and vol_expansion and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + bearish momentum + vol expansion + volume spike
            elif downtrend and rsi_bearish and vol_expansion and vol_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend reversal or momentum deterioration
            if not uptrend or not rsi_bullish:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend reversal or momentum deterioration
            if not downtrend or not rsi_bearish:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1dSMA50_ATRRegime_RSI_Volume_Spike_v1"
timeframe = "12h"
leverage = 1.0