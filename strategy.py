#!/usr/bin/env python3
# 4h_donchian_breakout_volume_atr_v1
# Hypothesis: 4h Donchian breakout strategy with volume confirmation and ATR-based stoploss.
# Long when price breaks above Donchian(20) high with volume > 1.5x 20-period average.
# Short when price breaks below Donchian(20) low with volume > 1.5x 20-period average.
# Exit via ATR trailing stop (3x ATR) or opposite breakout.
# Uses 1d HMA as higher timeframe trend filter: only long when price > 1d HMA, short when price < 1d HMA.
# Discrete position sizing (0.25) to minimize fee churn. Target: 20-50 trades/year.
# Works in bull/bear markets: Donchian captures breakouts, volume confirms conviction, 1d HMA avoids counter-trend trades.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # ATR (14-period) for trailing stop
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr_series = pd.Series(tr)
    atr = tr_series.rolling(window=14, min_periods=14).mean().values
    
    # Multi-timeframe: 1d HMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    close_1d_s = pd.Series(close_1d)
    # HMA(21) calculation
    n_half = int(21 / 2)
    n_sqrt = int(np.sqrt(21))
    wma_half = close_1d_s.rolling(window=n_half, min_periods=n_half).apply(
        lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True
    ).values
    wma_full = close_1d_s.rolling(window=21, min_periods=21).apply(
        lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True
    ).values
    raw_hma = 2 * wma_half - wma_full
    hma_1d = pd.Series(raw_hma).rolling(window=n_sqrt, min_periods=n_sqrt).apply(
        lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True
    ).values
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_long = 0.0
    lowest_since_short = 0.0
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr[i]) or
            np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(hma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Update highest high since entry
            if close[i] > highest_since_long:
                highest_since_long = close[i]
            # ATR trailing stop: exit if price drops 3*ATR from highest
            if close[i] < highest_since_long - 3.0 * atr[i]:
                position = 0
                highest_since_long = 0.0
                signals[i] = 0.0
            # Opposite breakout exit
            elif close[i] < donchian_low[i]:
                position = 0
                highest_since_long = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if close[i] < lowest_since_short:
                lowest_since_short = close[i]
            # ATR trailing stop: exit if price rises 3*ATR from lowest
            if close[i] > lowest_since_short + 3.0 * atr[i]:
                position = 0
                lowest_since_short = 0.0
                signals[i] = 0.0
            # Opposite breakout exit
            elif close[i] > donchian_high[i]:
                position = 0
                lowest_since_short = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for breakout with volume confirmation and 1d HMA filter
            bullish_breakout = (close[i] > donchian_high[i] and close[i-1] <= donchian_high[i-1]) and \
                              volume_confirmed and (close[i] > hma_1d_aligned[i])
            bearish_breakout = (close[i] < donchian_low[i] and close[i-1] >= donchian_low[i-1]) and \
                              volume_confirmed and (close[i] < hma_1d_aligned[i])
            
            if bullish_breakout:
                position = 1
                highest_since_long = close[i]
                signals[i] = 0.25
            elif bearish_breakout:
                position = -1
                lowest_since_short = close[i]
                signals[i] = -0.25
    
    return signals