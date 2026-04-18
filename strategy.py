#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR-based volatility filter
    df_1d = get_htf_data(prices, '1d')
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR(14) for volatility regime filter
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 4-hour ATR(10) for entry filter and position sizing
    tr1_4h = high[1:] - low[1:]
    tr2_4h = np.abs(high[1:] - close[:-1])
    tr3_4h = np.abs(low[1:] - close[:-1])
    tr_4h = np.concatenate([[np.nan], np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))])
    atr_10 = pd.Series(tr_4h).rolling(window=10, min_periods=10).mean().values
    
    # Calculate daily ATR ratio (current ATR / 50-period average) for volatility regime
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_14 / atr_ma_50
    
    # Align daily ATR ratio to 4h timeframe (only use after daily ATR is calculated)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate 4-hour RSI(14) for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for ATR calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr_ratio_aligned[i]) or np.isnan(atr_10[i]) or 
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: only trade in normal to high volatility (avoid low vol chop)
        vol_regime = atr_ratio_aligned[i] > 0.8 and atr_ratio_aligned[i] < 3.0
        
        # Mean reversion signals with volatility-adjusted thresholds
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Dynamic position size based on volatility (inverse volatility scaling)
        base_size = 0.25
        vol_scaling = min(1.0, 1.0 / atr_ratio_aligned[i])  # reduce size in high vol
        position_size = base_size * vol_scaling
        position_size = max(0.10, min(position_size, 0.35))  # clamp to reasonable range
        
        if position == 0:
            # Long: RSI oversold + volatility regime
            if rsi_oversold and vol_regime:
                signals[i] = position_size
                position = 1
            # Short: RSI overbought + volatility regime
            elif rsi_overbought and vol_regime:
                signals[i] = -position_size
                position = -1
        
        elif position == 1:
            # Long exit: RSI overbought or volatility too low (chop) or too high
            if rsi_overbought or not vol_regime:
                signals[i] = -position_size  # reverse to short
                position = -1
            else:
                signals[i] = position_size
        
        elif position == -1:
            # Short exit: RSI oversold or volatility too low (chop) or too high
            if rsi_oversold or not vol_regime:
                signals[i] = position_size  # reverse to long
                position = 1
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_RSI_Volatility_MeanReversion"
timeframe = "4h"
leverage = 1.0