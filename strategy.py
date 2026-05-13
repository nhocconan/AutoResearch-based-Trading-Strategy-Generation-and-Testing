#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and ATR(14) > ATR(50) volatility regime.
# Long when price breaks above Donchian upper channel AND 1w EMA34 up-trend AND high volatility regime.
# Short when price breaks below Donchian lower channel AND 1w EMA34 down-trend AND high volatility regime.
# Exit when price crosses Donchian midline (mean of upper/lower) OR volatility regime shifts to low vol.
# Uses discrete position sizing (0.25) to limit fee churn and manage drawdown.
# Designed for low trade frequency (~7-25/year) by requiring confluence of breakout, weekly trend, and volatility regime.
# Donchian channels provide clear breakout levels; weekly EMA filter ensures alignment with higher-timeframe trend;
# volatility regime filters out low-conviction choppy markets. Works in both bull and bear by capturing strong
# directional moves with trend and volatility confirmation.

name = "1d_Donchian20_Breakout_1wTrend_VolRegime_v1"
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
    
    # Calculate EMA(34) on 1w close for trend filter (more responsive than EMA50 for daily)
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Donchian(20) channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # ATR(14) and ATR(50) for volatility regime
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]  # first bar
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Volatility regime: ATR14 > ATR50 (high volatility)
    vol_regime = atr14 > atr50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(ema34_1w_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or \
           np.isnan(atr14[i]) or np.isnan(atr50[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper channel AND 1w EMA34 up-trend AND high volatility regime
            if close[i] > highest_high[i] and ema34_1w_aligned[i] > ema34_1w_aligned[i-1] and vol_regime[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower channel AND 1w EMA34 down-trend AND high volatility regime
            elif close[i] < lowest_low[i] and ema34_1w_aligned[i] < ema34_1w_aligned[i-1] and vol_regime[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses Donchian midline OR volatility regime shifts to low vol
            if close[i] < donchian_mid[i] or not vol_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses Donchian midline OR volatility regime shifts to low vol
            if close[i] > donchian_mid[i] or not vol_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals