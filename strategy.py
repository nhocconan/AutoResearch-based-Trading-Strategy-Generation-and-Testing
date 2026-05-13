#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray Bull/Bear Power with 1d EMA34 trend filter and volatility-adjusted position sizing.
# Elder Ray: Bull Power = High - EMA13(close), Bear Power = Low - EMA13(close).
# Long when Bull Power > 0 AND Bear Power rising (less negative) AND price > 1d EMA34.
# Short when Bear Power < 0 AND Bull Power falling (less positive) AND price < 1d EMA34.
# Position size scaled by ATR regime: 0.30 in high volatility (ATR14 > ATR50), 0.15 in low volatility.
# Uses discrete levels to minimize fee churn. Designed for 12-37 trades/year by requiring EMA13 alignment and daily trend filter.
# Works in bull markets via Bull Power strength and in bear markets via Bear Power exhaustion signals.

name = "6h_ElderRay_1dTrend_VolRegime_v1"
timeframe = "6h"
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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(34) on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Elder Ray components
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # Slope of Bull/Bear Power (1-bar change) to detect rising/falling momentum
    bull_power_slope = bull_power - np.roll(bull_power, 1)
    bear_power_slope = bear_power - np.roll(bear_power, 1)
    bull_power_slope[0] = 0
    bear_power_slope[0] = 0
    
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
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(ema13[i]) or np.isnan(bull_power[i]) or \
           np.isnan(bear_power[i]) or np.isnan(bull_power_slope[i]) or np.isnan(bear_power_slope[i]) or \
           np.isnan(atr14[i]) or np.isnan(atr50[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power > 0 (strong bullish momentum) AND Bear Power rising (less negative) AND price > 1d EMA34
            if bull_power[i] > 0 and bear_power_slope[i] > 0 and close[i] > ema34_1d_aligned[i]:
                signals[i] = position_size[i]
                position = 1
            # SHORT: Bear Power < 0 (strong bearish momentum) AND Bull Power falling (less positive) AND price < 1d EMA34
            elif bear_power[i] < 0 and bull_power_slope[i] < 0 and close[i] < ema34_1d_aligned[i]:
                signals[i] = -position_size[i]
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull Power <= 0 (loss of bullish momentum) OR Bear Power <= 0 (selling pressure) OR price < 1d EMA34 (trend break)
            if bull_power[i] <= 0 or bear_power[i] <= 0 or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = position_size[i]
        elif position == -1:
            # EXIT SHORT: Bear Power >= 0 (loss of bearish momentum) OR Bull Power >= 0 (buying pressure) OR price > 1d EMA34 (trend break)
            if bear_power[i] >= 0 or bull_power[i] >= 0 or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -position_size[i]
    
    return signals