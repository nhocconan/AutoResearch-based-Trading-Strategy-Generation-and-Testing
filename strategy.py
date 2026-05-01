#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h/1d HTF for signal direction and 1h for entry timing.
# Uses 4h Supertrend for trend bias and 1d ATR-based volatility regime filter.
# Enters on 1h pullbacks to 20 EMA in direction of 4h trend when 1d volatility is elevated.
# Designed for low trade frequency: ~15-37 trades/year per symbol with 0.20 sizing.
# Works in bull markets by following 4h uptrend, in bear markets by following 4h downtrend.
# Volatility regime filter avoids choppy markets and captures expansion phases.

name = "1h_Supertrend4h_ATR1d_VolRegime_Pullback_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h HTF data for Supertrend trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 15:
        return np.zeros(n)
    
    # Calculate 4h Supertrend (ATR=10, mult=3.0)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr1[0] = tr2[0] = tr3[0] = 0.0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR
    atr_4h = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Supertrend
    hl2 = (high_4h + low_4h) / 2.0
    upper = hl2 + (3.0 * atr_4h)
    lower = hl2 - (3.0 * atr_4h)
    
    upper_band = np.zeros_like(upper)
    lower_band = np.zeros_like(lower)
    upper_band[0] = upper[0]
    lower_band[0] = lower[0]
    
    for i in range(1, len(close_4h)):
        if close_4h[i-1] <= upper_band[i-1]:
            upper_band[i] = min(upper[i], upper_band[i-1])
        else:
            upper_band[i] = upper[i]
        
        if close_4h[i-1] >= lower_band[i-1]:
            lower_band[i] = max(lower[i], lower_band[i-1])
        else:
            lower_band[i] = lower[i]
    
    supertrend = np.zeros_like(close_4h)
    direction = np.ones_like(close_4h)  # 1 for uptrend, -1 for downtrend
    for i in range(len(close_4h)):
        if i == 0:
            supertrend[i] = upper_band[i]
            direction[i] = 1
        else:
            if close_4h[i] <= upper_band[i]:
                supertrend[i] = upper_band[i]
                direction[i] = 1
            elif close_4h[i] >= lower_band[i]:
                supertrend[i] = lower_band[i]
                direction[i] = -1
            else:
                supertrend[i] = supertrend[i-1]
                direction[i] = direction[i-1]
    
    # 4h trend bias: 1 for uptrend, -1 for downtrend, 0 for undefined
    trend_bias_4h = direction
    trend_bias_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_bias_4h)
    
    # 1d HTF data for volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ATR ratio (ATR(7)/ATR(30)) for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr1_1d[0] = tr2_1d[0] = tr3_1d[0] = 0.0
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    
    # ATR(7) and ATR(30)
    atr_7_1d = pd.Series(tr_1d).ewm(span=7, adjust=False, min_periods=7).mean().values
    atr_30_1d = pd.Series(tr_1d).ewm(span=30, adjust=False, min_periods=30).mean().values
    
    # Avoid division by zero
    atr_ratio_1d = np.where(atr_30_1d > 0, atr_7_1d / atr_30_1d, 1.0)
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Volatility regime: elevated when ATR ratio > 1.2 (expansion phase)
    vol_regime = atr_ratio_1d_aligned > 1.2
    
    # 1h EMA20 for pullback entries
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need 4h Supertrend (15 bars) + 1h EMA20 (20 bars) + 1d ATR ratio (30 days)
    start_idx = max(15, 20, 30)
    
    for i in range(start_idx, n):
        if (np.isnan(trend_bias_4h_aligned[i]) or np.isnan(ema_20[i]) or 
            np.isnan(atr_ratio_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Only trade during session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 4h Supertrend
        bullish_trend = trend_bias_4h_aligned[i] == 1
        bearish_trend = trend_bias_4h_aligned[i] == -1
        
        # Volatility regime filter
        high_vol = vol_regime[i]
        
        if position == 0:  # Flat - look for new entries
            if bullish_trend and high_vol:
                # Long: pullback to EMA20 in uptrend
                if close[i] <= ema_20[i] * 1.005:  # within 0.5% above EMA20
                    signals[i] = 0.20
                    position = 1
                else:
                    signals[i] = 0.0
            elif bearish_trend and high_vol:
                # Short: pullback to EMA20 in downtrend
                if close[i] >= ema_20[i] * 0.995:  # within 0.5% below EMA20
                    signals[i] = -0.20
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid chop or low volatility
        
        elif position == 1:  # Long position
            # Exit: trend reversal or volatility collapse
            if not bullish_trend or not high_vol:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: trend reversal or volatility collapse
            if not bearish_trend or not high_vol:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals