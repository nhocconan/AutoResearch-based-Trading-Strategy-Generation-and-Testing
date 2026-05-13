#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND price > 1d EMA34 AND volume > 1.5x volume MA(20).
# Short when price breaks below Donchian(20) low AND price < 1d EMA34 AND volume > 1.5x volume MA(20).
# Uses ATR(14) for volatility-adjusted position sizing: 0.30 in high volatility (ATR14 > ATR50), 0.15 in low volatility.
# Discrete position sizes to minimize fee churn. Designed for 19-50 trades/year by requiring confluence of trend, breakout, and volume.
# Works in bull markets via breakout strength and in bear markets via short-side breakouts with trend filter.

name = "4h_Donchian20_1dTrend_Volume_VolRegime_v1"
timeframe = "4h"
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
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(34) on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x volume MA(20)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * volume_ma)
    
    # ATR(14) and ATR(50) for volatility regime
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]  # first bar
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Volatility regime: ATR14 > ATR50 (high volatility) -> full size, else half size
    vol_regime = atr14 > atr50
    position_size = np.where(vol_regime, 0.30, 0.15)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or \
           np.isnan(volume_ma[i]) or np.isnan(atr14[i]) or np.isnan(atr50[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high AND price > 1d EMA34 AND volume confirmation
            if close[i] > highest_high[i] and close[i] > ema34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = position_size[i]
                position = 1
            # SHORT: Price breaks below Donchian low AND price < 1d EMA34 AND volume confirmation
            elif close[i] < lowest_low[i] and close[i] < ema34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -position_size[i]
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low OR price < 1d EMA34 (trend break)
            if close[i] < lowest_low[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = position_size[i]
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high OR price > 1d EMA34 (trend break)
            if close[i] > highest_high[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -position_size[i]
    
    return signals