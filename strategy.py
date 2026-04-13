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
    
    # 1-day data for ATR and volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 4-hour ATR for volatility filter
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, tr2)])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily ATR for volatility regime filter
    tr1d = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_d = np.concatenate([[np.inf], np.maximum(tr1d, tr2d)])
    atr_1d = pd.Series(tr_d).rolling(window=14, min_periods=14).mean().values
    atr_1d_ma = pd.Series(atr_1d).rolling(window=10, min_periods=10).mean().values
    
    # Align daily ATR and MA to 4h
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_1d_ma_aligned = align_htf_to_ltf(prices, df_1d, atr_1d_ma)
    
    # 4-hour Bollinger Bands (20, 2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean()
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std()
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    
    # Bollinger Band Width for squeeze detection
    bb_width = (upper_bb - lower_bb) / sma_20
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(atr[i]) or np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or
            np.isnan(bb_width[i]) or np.isnan(bb_width_ma[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(atr_1d_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: only trade when volatility is low (squeeze)
        # BB Width < MA of BB Width indicates low volatility/squeeze
        volatility_low = bb_width[i] < bb_width_ma[i]
        
        # Additional volatility filter: daily ATR below its MA (low volatility environment)
        vol_regime = atr_1d_aligned[i] < atr_1d_ma_aligned[i]
        
        # Only trade in low volatility regimes
        if not (volatility_low and vol_regime):
            signals[i] = 0.0
            continue
        
        # Bollinger Band breakout signals
        # Long: price breaks above upper BB with volume confirmation
        # Short: price breaks below lower BB with volume confirmation
        long_breakout = close[i] > upper_bb[i]
        short_breakout = close[i] < lower_bb[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        # Use 20-period volume average for confirmation
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        if np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        volume_confirm = volume[i] > (vol_ma[i] * 1.5)
        
        # Entry conditions
        if position == 0:
            if long_breakout and volume_confirm:
                position = 1
                signals[i] = position_size
            elif short_breakout and volume_confirm:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when price returns to middle of Bollinger Bands
            middle_bb = sma_20.iloc[i] if hasattr(sma_20, 'iloc') else sma_20[i]
            if close[i] < middle_bb:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when price returns to middle of Bollinger Bands
            middle_bb = sma_20.iloc[i] if hasattr(sma_20, 'iloc') else sma_20[i]
            if close[i] > middle_bb:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_BB_Squeeze_Breakout_Volume"
timeframe = "4h"
leverage = 1.0