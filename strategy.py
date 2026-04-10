#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power + 1d Regime Filter
# - Uses Elder Ray Bull/Bear Power (EMA13 of high/low) to measure buying/selling pressure
# - 1d ADX regime filter: only take longs when ADX>25 AND +DI>-DI (strong uptrend)
# - Only take shorts when ADX>25 AND -DI>+DI (strong downtrend)
# - Avoids whipsaws in ranging markets by requiring strong trend confirmation
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 12-30 trades/year on 6h timeframe (50-120 total over 4 years)
# - Elder Ray works well in trending markets; 1d ADX filter ensures we only trade with the higher timeframe trend

name = "6h_1d_elder_ray_power_regime_v1"
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
    
    # Pre-compute 1d ADX and DI for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    # Calculate +DM and -DM
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (EMA with alpha=1/period)
    atr_period = 14
    alpha = 1.0 / atr_period
    
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]
    
    plus_dm_smooth = np.zeros_like(plus_dm)
    minus_dm_smooth = np.zeros_like(minus_dm)
    plus_dm_smooth[0] = plus_dm[0]
    minus_dm_smooth[0] = minus_dm[0]
    for i in range(1, len(plus_dm)):
        plus_dm_smooth[i] = alpha * plus_dm[i] + (1 - alpha) * plus_dm_smooth[i-1]
        minus_dm_smooth[i] = alpha * minus_dm[i] + (1 - alpha) * minus_dm_smooth[i-1]
    
    # Calculate +DI and -DI
    plus_di = 100 * plus_dm_smooth / (atr + 1e-10)
    minus_di = 100 * minus_dm_smooth / (atr + 1e-10)
    
    # Calculate ADX (smoothed average of |+DI - -DI|)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = np.zeros_like(dx)
    adx[atr_period-1] = np.mean(dx[atr_period-1:2*(atr_period-1)+1]) if len(dx) >= 2*atr_period-1 else dx[0]
    for i in range(atr_period, len(dx)):
        adx[i] = (alpha * dx[i]) + ((1 - alpha) * adx[i-1])
    
    # Align 1d regime indicators to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    plus_di_aligned = align_htf_to_ltf(prices, df_1d, plus_di)
    minus_di_aligned = align_htf_to_ltf(prices, df_1d, minus_di)
    
    # Pre-compute Elder Ray Power on 6d data (using 13-period EMA of close)
    ema13 = prices['close'].ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = prices['high'].values - ema13  # Bull Power = High - EMA13
    bear_power = ema13 - prices['low'].values  # Bear Power = EMA13 - Low
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or np.isnan(plus_di_aligned[i]) or 
            np.isnan(minus_di_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries with regime filter
            # Long when Bull Power > 0 (buying pressure) AND strong uptrend regime (ADX>25 and +DI>-DI)
            if (bull_power[i] > 0 and 
                adx_aligned[i] > 25 and 
                plus_di_aligned[i] > minus_di_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short when Bear Power > 0 (selling pressure) AND strong downtrend regime (ADX>25 and -DI>+DI)
            elif (bear_power[i] > 0 and 
                  adx_aligned[i] > 25 and 
                  minus_di_aligned[i] > plus_di_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit when power fades or regime changes
            # Exit conditions:
            # 1. Power fails (Bull Power <= 0 for longs, Bear Power <= 0 for shorts)
            # 2. Regime weakens (ADX < 20) - trend exhaustion
            # 3. Regime reverses (opposite DI crossover)
            exit_signal = False
            
            if position == 1:  # Long position
                if (bull_power[i] <= 0 or  # Buying pressure gone
                    adx_aligned[i] < 20 or  # Trend too weak
                    minus_di_aligned[i] > plus_di_aligned[i]):  # Regime turned bearish
                    exit_signal = True
            elif position == -1:  # Short position
                if (bear_power[i] <= 0 or   # Selling pressure gone
                    adx_aligned[i] < 20 or   # Trend too weak
                    plus_di_aligned[i] > minus_di_aligned[i]):  # Regime turned bullish
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals