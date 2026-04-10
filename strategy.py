#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Bull/Bear Power with 1w trend filter and ATR volatility regime
# - Bull Power = High - EMA13(close), Bear Power = EMA13(close) - Low
# - Long when Bull Power > 0 AND Bear Power < 0 (bullish momentum) in 1w uptrend (close > EMA34) with ATR(14) > ATR(50) (expanding volatility)
# - Short when Bear Power > 0 AND Bull Power < 0 (bearish momentum) in 1w downtrend (close < EMA34) with ATR(14) > ATR(50)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets ~20-30 trades/year (80-120 total over 4 years) to avoid fee drag
# - 1w trend filter ensures alignment with major market structure
# - ATR regime filter avoids choppy markets where Elder Ray gives false signals
# - Works in both bull and bear markets by following 1w trend

name = "6h_1w_elderray_atrregime_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w indicators
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # 1w EMA(34) for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Pre-compute 6h indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 6h EMA(13) for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    # 6h ATR(14) and ATR(50) for volatility regime
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Volatility regime: expanding (ATR14 > ATR50)
    vol_expanding = atr_14 > atr_50
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(atr_14[i]) or np.isnan(atr_50[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: Bullish momentum in 1w uptrend with expanding volatility
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                close[i] > ema_34_1w_aligned[i] and vol_expanding[i]):
                position = 1
                signals[i] = 0.25
            # Short signal: Bearish momentum in 1w downtrend with expanding volatility
            elif (bear_power[i] > 0 and bull_power[i] < 0 and 
                  close[i] < ema_34_1w_aligned[i] and vol_expanding[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit when momentum diverges or volatility contracts
            if position == 1:  # Long
                if (bull_power[i] <= 0 or bear_power[i] >= 0 or not vol_expanding[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # Short
                if (bear_power[i] <= 0 or bull_power[i] >= 0 or not vol_expanding[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals