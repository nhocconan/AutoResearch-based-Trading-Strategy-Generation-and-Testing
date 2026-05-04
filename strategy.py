#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation
# Donchian channel breakouts capture strong momentum moves. In bull markets, buying
# upper channel breakouts in uptrends works; in bear markets, selling lower channel
# breakdowns in downtrends works. The 1d ATR regime filter avoids whipsaws in low-vol
# ranging markets by only allowing trades when volatility is elevated (ATR > its 50-period MA).
# Volume confirmation (1.5x 20-period EMA) ensures breakouts have participation.
# Designed for 4h timeframe to target 20-50 trades/year (75-200 total over 4 years) with
# discrete sizing (0.30). Uses 1d ATR for smoother regime filter that adapts to changing
# volatility regimes, improving performance in both trending and ranging markets.

name = "4h_Donchian20_1dATR_Regime_Volume_Confirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d True Range and ATR(50) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # ATR(50) using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    atr_50_1d = pd.Series(tr).ewm(alpha=1/50, adjust=False, min_periods=50).mean().values
    atr_ma_50_1d = pd.Series(atr_50_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align ATR and its MA to 4h timeframe
    atr_50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_50_1d)
    atr_ma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_50_1d)
    
    # Calculate 4h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_high = high_series.rolling(window=20, min_periods=20).max().values
    donch_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 1.5x 20-period EMA on 4h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(atr_50_1d_aligned[i]) or np.isnan(atr_ma_50_1d_aligned[i]) or
            np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: 1d ATR > its 50-period MA (elevated volatility)
        volatility_regime = atr_50_1d_aligned[i] > atr_ma_50_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirmed = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            # Long: close breaks above Donchian high + volume confirmation + volatility regime
            if (close[i] > donch_high[i] and volume_confirmed and volatility_regime):
                signals[i] = 0.30
                position = 1
            # Short: close breaks below Donchian low + volume confirmation + volatility regime
            elif (close[i] < donch_low[i] and volume_confirmed and volatility_regime):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price falls below Donchian low (mean reversion) OR volatility regime ends
            if close[i] < donch_low[i] or not volatility_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price rises above Donchian high (mean reversion) OR volatility regime ends
            if close[i] > donch_high[i] or not volatility_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals