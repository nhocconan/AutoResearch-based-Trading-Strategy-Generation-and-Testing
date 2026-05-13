#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray Power (Bull/Bear) with 12h trend filter (EMA34), volume confirmation (2x MA20), and Bollinger Band squeeze release.
# Elder Ray measures bull/bear power via EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13.
# Enters long when Bull Power > 0 AND 12h bullish trend (close > EMA34) AND volume > 2x MA20 AND BBWidth < 0.05 (squeeze) AND BBWidth increasing (breakout).
# Enters short when Bear Power < 0 AND 12h bearish trend (close < EMA34) AND volume > 2x MA20 AND BBWidth < 0.05 (squeeze) AND BBWidth increasing (breakout).
# Exits when Elder Ray power reverses (Bull Power < 0 for long, Bear Power > 0 for short) OR price reaches opposite Bollinger Band (mean reversion).
# Uses discrete position sizing (0.25) to limit fee churn and manage drawdown.
# Designed for low trade frequency (~12-37/year) by requiring confluence: Elder Ray alignment + HTF trend + volume spike + volatility expansion from squeeze.
# Works in bull markets (captures strong trends via Elder Ray) and bear markets (avoids false signals via 12h trend filter and squeeze condition).

name = "6h_ElderRay_Power_12hTrend_Volume_SqueezeBreakout_v1"
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
    volume = prices['volume'].values
    
    # Calculate EMA13 for Elder Ray Power
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # Get 12h data for trend filter (EMA34)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Volume filter: current volume > 2x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)
    
    # Bollinger Bands (20, 2) for squeeze detection
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma20 + (2 * std20)
    lower_band = sma20 - (2 * std20)
    bb_width = (upper_band - lower_band) / sma20  # Normalized BB width
    bb_width_prev = np.roll(bb_width, 1)
    bb_width_prev[0] = bb_width[0]  # first bar
    bb_width_expanding = bb_width > bb_width_prev  # BB width increasing (squeeze release)
    bb_squeeze = bb_width < 0.05  # Squeeze condition (low volatility)
    
    # Track entry price for mean reversion exit
    entry_price = np.full(n, np.nan)
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or \
           np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_ma20[i]) or \
           np.isnan(sma20[i]) or np.isnan(std20[i]) or np.isnan(bb_width[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power > 0 AND 12h bullish trend AND volume spike AND BB squeeze release
            if bull_power[i] > 0 and close[i] > ema34_12h_aligned[i] and volume_spike[i] and bb_squeeze[i] and bb_width_expanding[i]:
                signals[i] = 0.25
                position = 1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            # SHORT: Bear Power < 0 AND 12h bearish trend AND volume spike AND BB squeeze release
            elif bear_power[i] < 0 and close[i] < ema34_12h_aligned[i] and volume_spike[i] and bb_squeeze[i] and bb_width_expanding[i]:
                signals[i] = -0.25
                position = -1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Elder Ray power reverses (Bull Power < 0) OR price reaches lower Bollinger Band (mean reversion)
            if bull_power[i] < 0 or close[i] < lower_band[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.25
                entry_price[i] = entry_price[i-1]  # carry forward entry price
        elif position == -1:
            # EXIT SHORT: Elder Ray power reverses (Bear Power > 0) OR price reaches upper Bollinger Band (mean reversion)
            if bear_power[i] > 0 or close[i] > upper_band[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = -0.25
                entry_price[i] = entry_price[i-1]  # carry forward entry price
    
    return signals