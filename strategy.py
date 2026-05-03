#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull Power/Bear Power) with 1d EMA50 trend filter and ATR-based volume confirmation.
# Long when Bull Power > 0 (close > EMA13) AND Bear Power < 0 (low < EMA13) in bull trend (close > 1d EMA50) with volume > 1.5x ATR(20).
# Short when Bear Power < 0 (low < EMA13) AND Bull Power < 0 (close < EMA13) in bear trend (close < 1d EMA50) with volume spike.
# Uses discrete position sizing (0.25) to minimize fee churn.
# Elder Ray measures bull/bear power relative to EMA13, providing early trend strength signals.
# 1d EMA50 filter ensures alignment with higher timeframe trend to avoid whipsaws.
# Volume confirmation via ATR-normalized volume ensures institutional participation.
# Target: 50-150 total trades over 4 years (12-37/year) with Sharpe > 0 on BTC/ETH/SOL.

name = "6h_ElderRay_1dEMA50_ATRVOL"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate EMA13 for Elder Ray (6h)
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power: high - EMA13
    bear_power = low - ema_13   # Bear Power: low - EMA13
    
    # ATR(20) for volume normalization
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.roll(close, 1))).values
    tr3 = pd.Series(np.abs(low - np.roll(close, 1))).values
    tr2[0] = tr1[0]  # First bar: no previous close
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Volume regime: current volume > 1.5x ATR(20) * average volume ratio
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (1.5 * atr_20 * (vol_ma_50 / close))  # ATR-normalized volume spike
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(atr_20[i]) or np.isnan(vol_ma_50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_1d_aligned[i]
        bp = bull_power[i]
        br = bear_power[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Elder Ray conditions
        bull_strong = bp > 0  # Bull Power positive
        bear_strong = br < 0  # Bear Power negative
        
        # Entry logic
        if position == 0:
            if is_bull_trend and bull_strong and br < 0 and vol_spike:  # Bull trend, bulls in control, bears weak
                signals[i] = 0.25
                position = 1
            elif is_bear_trend and bear_strong and bp < 0 and vol_spike:  # Bear trend, bears in control, bulls weak
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bear Power turns positive OR trend reversal
            if br >= 0 or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bull Power turns negative OR trend reversal
            if bp <= 0 or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals