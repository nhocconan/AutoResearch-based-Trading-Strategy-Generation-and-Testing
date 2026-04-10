#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h TRIX(12) momentum with 1d volume spike and ADX(14) > 20 trend filter
# - Long when TRIX crosses above zero + 1d volume > 1.8x 20-period volume SMA + ADX > 20
# - Short when TRIX crosses below zero + 1d volume > 1.8x 20-period volume SMA + ADX > 20
# - Exit: TRIX returns to zero line (momentum fade)
# - Position sizing: 0.25 discrete level
# - TRIX filters noise better than MACD, volume confirms institutional interest, ADX avoids chop
# - Works in bull/bear: momentum divergences effective in both regimes when combined with volume
# - 12h timeframe targets 15-35 trades/year with strict entry conditions to minimize fee drag

name = "12h_1d_trix_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 12h TRIX(12,12,12) - triple smoothed EMA of ROC
    # ROC = (close - close[period]) / close[period] * 100
    roc_period = 12
    roc = np.zeros_like(close)
    roc[roc_period:] = (close[roc_period:] - close[:-roc_period]) / close[:-roc_period] * 100
    
    # Triple EMA smoothing
    ema1 = pd.Series(roc).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    
    # Calculate 12h ADX(14) for trend filter
    # True Range
    tr1 = np.maximum(high - low, 
                     np.maximum(np.abs(high - np.roll(close, 1)), 
                                np.abs(low - np.roll(close, 1))))
    tr1[0] = high[0] - low[0]
    # Plus Directional Movement
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    plus_dm[0] = 0
    # Minus Directional Movement
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    minus_dm[0] = 0
    # Smoothed values
    atr = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    # Handle division by zero and invalid values
    plus_di = np.where(atr == 0, 0, plus_di)
    minus_di = np.where(atr == 0, 0, minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = np.where(np.isnan(adx) | np.isinf(adx), 0, adx)
    
    # Calculate 1d volume SMA(20) for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(trix[i]) or np.isnan(trix[i-1]) or np.isnan(adx[i]) or 
            np.isnan(volume_sma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.8x 20-period SMA (volume spike)
        vol_1d_current = align_htf_to_ltf(prices, df_1d, df_1d['volume'].values)
        vol_confirm = vol_1d_current[i] > 1.8 * volume_sma_20_1d_aligned[i]
        
        # Trend filter: ADX > 20 indicates trending market (avoid choppy markets)
        trending_market = adx[i] > 20
        
        # TRIX zero-line crossover signals
        trix_now = trix[i]
        trix_prev = trix[i-1]
        trix_cross_above = (trix_prev <= 0) and (trix_now > 0)
        trix_cross_below = (trix_prev >= 0) and (trix_now < 0)
        
        # Entry conditions: TRIX zero-line cross with volume and trend confirmation
        long_entry = trix_cross_above and vol_confirm and trending_market
        short_entry = trix_cross_below and vol_confirm and trending_market
        
        # Exit conditions: TRIX returns to zero line (momentum fade)
        long_exit = trix_now < 0  # Exit long when TRIX goes negative
        short_exit = trix_now > 0  # Exit short when TRIX goes positive
        
        if position == 0:  # Flat - look for entry
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if long_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if short_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals