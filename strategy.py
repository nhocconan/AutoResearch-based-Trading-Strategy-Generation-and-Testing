#!/usr/bin/env python3
"""
Hypothesis: 1h RSI(2) mean reversion with 4h trend filter and 1d volatility regime.
- RSI(2) < 10 for long, > 90 for short - extreme short-term reversals
- 4h EMA50 as trend filter: only long when price > EMA50, short when price < EMA50
- 1d ATR ratio (ATR5/ATR30) < 0.8 for low volatility regime (mean reversion works better)
- Volume confirmation: current volume > 1.2x 20-period average
- Position size: 0.20 discrete level to minimize fee churn
- Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years)
- Works in bull/bear via 4h trend filter and volatility regime filter
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI(2) for extreme short-term mean reversion
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Volume confirmation: > 1.2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d data for ATR ratio volatility regime
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ATR5 and ATR30 on 1d
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr1 = np.maximum(tr1, np.abs(low_1d[1:] - close_1d[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])
    
    atr5 = pd.Series(tr1).rolling(window=5, min_periods=5).mean().values
    atr30 = pd.Series(tr1).rolling(window=30, min_periods=30).mean().values
    atr_ratio = atr5 / atr30  # Low when ATR5 < ATR30 (low volatility regime)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(2, 20, 50, 30)  # RSI2, volume MA, 4h EMA50, 1d ATR30
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(atr_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.2x average)
        volume_confirm = volume[i] > 1.2 * vol_ma[i]
        
        # Low volatility regime (ATR ratio < 0.8)
        low_vol_regime = atr_ratio_aligned[i] < 0.8
        
        if position == 0:
            # Long: RSI(2) < 10 AND price > 4h EMA50 AND low vol regime AND volume confirmation
            if rsi[i] < 10 and close[i] > ema_50_4h_aligned[i] and low_vol_regime and volume_confirm:
                signals[i] = 0.20
                position = 1
            # Short: RSI(2) > 90 AND price < 4h EMA50 AND low vol regime AND volume confirmation
            elif rsi[i] > 90 and close[i] < ema_50_4h_aligned[i] and low_vol_regime and volume_confirm:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: RSI(2) > 50 (mean reversion complete) OR volatility increases OR volume dries up
            if rsi[i] > 50 or atr_ratio_aligned[i] > 1.2 or volume[i] < 0.8 * vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: RSI(2) < 50 (mean reversion complete) OR volatility increases OR volume dries up
            if rsi[i] < 50 or atr_ratio_aligned[i] > 1.2 or volume[i] < 0.8 * vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI2_MeanReversion_4hEMA50_1dATRRatio_v1"
timeframe = "1h"
leverage = 1.0