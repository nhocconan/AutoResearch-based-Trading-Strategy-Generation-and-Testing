#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Keltner Channel breakout with 1-day trend filter and volume confirmation
# Keltner Channel (EMA20 ± 2*ATR10) identifies volatility-based breakouts
# In bull market (price > 1-day EMA50): buy upper band breakout, sell lower band breakout
# In bear market (price < 1-day EMA50): sell upper band breakout, buy lower band breakout
# Volume confirmation: require volume > 1.5x 20-period average
# Designed to capture breakouts in trending markets while filtering false signals in ranges
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 50-period EMA on daily timeframe for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate ATR for Keltner Channel (using 10-period ATR)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # True Range calculation
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # ATR(10)
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # EMA(20) for Keltner Channel middle line
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channel bands
    kc_upper = ema20 + 2 * atr
    kc_lower = ema20 - 2 * atr
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(ema20[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market trend
        is_bull = close[i] > ema50_1d_aligned[i]
        is_bear = close[i] < ema50_1d_aligned[i]
        
        # Volume confirmation
        has_volume = vol_filter[i]
        
        price = close[i]
        upper_band = kc_upper[i]
        lower_band = kc_lower[i]
        
        if position == 0:
            # Enter long: upper band breakout in bull OR lower band breakout in bear
            long_signal = False
            if has_volume:
                if (is_bull and price > upper_band) or (is_bear and price > upper_band):
                    long_signal = True
            
            # Enter short: lower band breakout in bull OR upper band breakout in bear
            short_signal = False
            if has_volume:
                if (is_bull and price < lower_band) or (is_bear and price < lower_band):
                    short_signal = True
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below middle line (EMA20)
            exit_signal = False
            if price < ema20[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above middle line (EMA20)
            exit_signal = False
            if price > ema20[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KeltnerBreakout_TrendFilter_Volume"
timeframe = "12h"
leverage = 1.0