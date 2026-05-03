#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d EMA50 trend filter + ATR regime filter
# Elder Ray measures bull/bear power relative to EMA13; trend filter ensures we trade with higher timeframe direction
# ATR regime filter avoids ranging markets (low ATR) and extreme volatility (high ATR)
# Works in bull/bear: 1d EMA50 defines major trend, Elder Ray captures momentum within that trend
# Target: 12-30 trades/year (50-120 total over 4 years) to minimize fee drag

name = "6h_ElderRay_1dEMA50_ATRRegime"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for regime filter on 6h data
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ATR regime: avoid low volatility (<0.5*20-period mean) and extreme volatility (>2.0*20-period mean)
    atr_ma = pd.Series(atr).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_regime = (atr > 0.5 * atr_ma) & (atr < 2.0 * atr_ma)
    
    # Calculate EMA(13) for Elder Ray on 6h data
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(atr_regime[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only trade in moderate volatility
        if not atr_regime[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Elder Ray signals with 1d trend filter
        # Long: Bull Power > 0 (bulls in control) + price above 1d EMA50
        # Short: Bear Power < 0 (bears in control) + price below 1d EMA50
        if position == 0:
            if bull_power[i] > 0 and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            elif bear_power[i] < 0 and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 (bulls lose control) OR price below 1d EMA50
            if bull_power[i] <= 0 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power >= 0 (bears lose control) OR price above 1d EMA50
            if bear_power[i] >= 0 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals