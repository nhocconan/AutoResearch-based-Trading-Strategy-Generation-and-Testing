#!/usr/bin/env python3
# Hypothesis: 1d Bollinger Band squeeze breakout with 1w EMA200 trend filter and ATR volatility regime.
# Long when price breaks above upper BB(20,2) AND close > 1w EMA200 AND ATR14 > ATR50 (high vol regime).
# Short when price breaks below lower BB(20,2) AND close < 1w EMA200 AND ATR14 > ATR50.
# Exit when price returns to middle BB(20) OR ATR14 < ATR50 * 0.8 (low vol exit).
# Uses discrete position sizing (0.30) to balance return and drawdown.
# Designed for low trade frequency (~7-25/year) by requiring confluence of BB breakout, weekly trend, and volatility regime.
# Bollinger Band squeeze identifies low volatility periods preceding breakouts, effective in both bull and bear markets.

name = "1d_BollingerBand_Squeeze_1wTrend_VolRegime_v1"
timeframe = "1d"
leverage = 1.0

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
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA(200) on 1w close for trend filter
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Bollinger Bands: middle = SMA(20), std = 2
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma20 + 2 * std20
    lower_band = sma20 - 2 * std20
    middle_band = sma20  # for exit
    
    # ATR(14) and ATR(50) for volatility regime
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]  # first bar
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Volatility regime: ATR14 > ATR50 (high volatility)
    vol_regime = atr14 > atr50
    
    # Breakout conditions
    breakout_up = close > upper_band
    breakout_down = close < lower_band
    return_to_middle = np.abs(close - middle_band) < 0.5 * std20  # exit when near middle
    
    # Track position
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(ema200_1w_aligned[i]) or np.isnan(sma20[i]) or np.isnan(std20[i]) or \
           np.isnan(atr14[i]) or np.isnan(atr50[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above upper BB, close > 1w EMA200, high volatility regime
            if breakout_up[i] and close[i] > ema200_1w_aligned[i] and vol_regime[i]:
                signals[i] = 0.30
                position = 1
            # SHORT: price breaks below lower BB, close < 1w EMA200, high volatility regime
            elif breakout_down[i] and close[i] < ema200_1w_aligned[i] and vol_regime[i]:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price returns to middle BB OR low volatility regime (ATR14 < ATR50 * 0.8)
            if return_to_middle[i] or atr14[i] < atr50[i] * 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: price returns to middle BB OR low volatility regime (ATR14 < ATR50 * 0.8)
            if return_to_middle[i] or atr14[i] < atr50[i] * 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals