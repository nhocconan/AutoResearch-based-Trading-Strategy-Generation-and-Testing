#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d ATR filter and volume confirmation
# Long when price breaks above 20-period Donchian high + ATR(1d) > 20-period mean ATR + volume > 1.5x 20-period volume SMA
# Short when price breaks below 20-period Donchian low + ATR(1d) > 20-period mean ATR + volume > 1.5x 20-period volume SMA
# Uses 1d ATR for volatility regime filter (only trade when volatility is elevated)
# Donchian breakouts capture strong trends; ATR filter avoids low-volatility false breakouts
# Volume confirmation reduces false signals
# Discrete position sizing (0.25) to control drawdown and minimize fee drag
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicator: ATR (20-period) and its mean for volatility regime ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar: no previous close
    tr2[0] = np.abs(high_1d[0] - close_1d[0])  # first bar: use same bar close
    tr3[0] = np.abs(low_1d[0] - close_1d[0])   # first bar: use same bar close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(20)
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # 20-period mean ATR for volatility regime filter
    mean_atr = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    
    # Align ATR and mean ATR to 12h timeframe
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    mean_atr_aligned = align_htf_to_ltf(prices, df_1d, mean_atr)
    
    # === 12h Indicators: Donchian Channel (20-period) ===
    # Donchian high: highest high over 20 periods
    # Donchian low: lowest low over 20 periods
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume SMA for confirmation (20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    # Need Donchian(20) + volume(20) + ATR(20) + mean ATR(20) + buffer
    warmup = 40
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(atr_aligned[i]) or np.isnan(mean_atr_aligned[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: current ATR > mean ATR (elevated volatility regime)
        vol_regime = atr_aligned[i] > mean_atr_aligned[i]
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Donchian high (20-period breakout)
        # 2. Elevated volatility regime (ATR > mean ATR)
        # 3. Volume confirmation
        if (close[i] > donchian_high[i]) and \
           vol_regime and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Donchian low (20-period breakout)
        # 2. Elevated volatility regime (ATR > mean ATR)
        # 3. Volume confirmation
        elif (close[i] < donchian_low[i]) and \
             vol_regime and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Donchian20_1dATR_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0