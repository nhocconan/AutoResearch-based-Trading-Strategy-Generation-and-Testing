#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1d trend filter and volume confirmation
# - Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# - Long when Bull Power > 0 AND Bear Power < 0 (strong bullish momentum) AND 1d EMA(50) > EMA(200) (bullish trend) AND 6h volume > 1.5x 20-bar avg
# - Short when Bear Power > 0 AND Bull Power < 0 (strong bearish momentum) AND 1d EMA(50) < EMA(200) (bearish trend) AND 6h volume > 1.5x 20-bar avg
# - Exit when Elder Ray shows weakening momentum (|Bull Power| < 0.1 * ATR AND |Bear Power| < 0.1 * ATR)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Elder Ray measures bull/bear power relative to EMA; 1d EMA filter ensures alignment with higher timeframe trend
# - Volume confirmation avoids low-liquidity false signals
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Works in both bull and bear markets: trend filter prevents counter-trend trades, volume confirms conviction

name = "6h_1d_elder_ray_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d EMA trend filter: EMA(50) vs EMA(200)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_bullish_1d = ema_50_1d > ema_200_1d
    ema_bearish_1d = ema_50_1d < ema_200_1d
    
    # Align 1d EMA trend to 6h timeframe
    ema_bullish_aligned = align_htf_to_ltf(prices, df_1d, ema_bullish_1d)
    ema_bearish_aligned = align_htf_to_ltf(prices, df_1d, ema_bearish_1d)
    
    # Pre-compute Elder Ray Index on 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # EMA(13) for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Bull Power = High - EMA(13)
    bull_power = high - ema_13
    # Bear Power = EMA(13) - Low
    bear_power = ema_13 - low
    
    # ATR(14) for exit condition
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.roll(close, 1))).values
    tr3 = pd.Series(np.abs(low - np.roll(close, 1))).values
    tr2[0] = tr1[0]  # first bar
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Elder Ray conditions: strong bull/bear power
    strong_bull = bull_power > 0
    strong_bear = bear_power > 0
    
    # Exit when momentum weakens (both powers near zero)
    weak_momentum = (np.abs(bull_power) < 0.1 * atr) & (np.abs(bear_power) < 0.1 * atr)
    
    # Pre-compute 6h volume confirmation: > 1.5x 20-period average
    volume = prices['volume'].values
    volume_20_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_bullish_aligned[i]) or np.isnan(ema_bearish_aligned[i]) or
            np.isnan(strong_bull[i]) or np.isnan(strong_bear[i]) or
            np.isnan(weak_momentum[i]) or np.isnan(vol_spike[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long when strong bull power AND 1d bullish trend AND volume spike
            if (strong_bull[i] and 
                ema_bullish_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short when strong bear power AND 1d bearish trend AND volume spike
            elif (strong_bear[i] and 
                  ema_bearish_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit on weakening momentum
            # Exit when Elder Ray shows weakening momentum
            exit_signal = weak_momentum[i]
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals