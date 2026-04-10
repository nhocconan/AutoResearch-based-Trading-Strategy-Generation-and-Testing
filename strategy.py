#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA filter and ATR-based volatility regime
# - Elder Ray Bull Power = High - EMA(13), Bear Power = Low - EMA(13) on 6h
# - Long when Bull Power > 0 AND Bear Power rising (less negative) AND 1d EMA(50) > EMA(200) AND ATR(14) < ATR(50) (low vol regime)
# - Short when Bear Power < 0 AND Bull Power falling (less positive) AND 1d EMA(50) < EMA(200) AND ATR(14) < ATR(50)
# - Exit when power reverses or ATR regime shifts to high volatility
# - 1d EMA filter ensures alignment with higher timeframe trend
# - ATR regime filter avoids whipsaws in high volatility periods
# - Target: 12-37 trades/year on 6h (50-150 total over 4 years) to avoid fee drag

name = "6h_1d_elderray_power_atr_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Pre-compute 1d EMA(50) and EMA(200)
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Pre-compute 6h indicators
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    vol_6h = prices['volume'].values
    
    # Elder Ray: EMA(13) on 6h
    ema_13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_6h - ema_13_6h  # Bull Power = High - EMA
    bear_power = low_6h - ema_13_6h   # Bear Power = Low - EMA
    
    # ATR(14) and ATR(50) for volatility regime
    tr1 = np.abs(high_6h[1:] - low_6h[1:])
    tr2 = np.abs(high_6h[1:] - close_6h[:-1])
    tr3 = np.abs(low_6h[1:] - close_6h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr_14 = np.full_like(tr, np.nan, dtype=float)
    atr_50 = np.full_like(tr, np.nan, dtype=float)
    
    if len(tr) >= 14:
        atr_14[13] = np.nanmean(tr[1:14])
        for i in range(14, len(tr)):
            if not np.isnan(tr[i]) and not np.isnan(atr_14[i-1]):
                atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    if len(tr) >= 50:
        atr_50[49] = np.nanmean(tr[1:50])
        for i in range(50, len(tr)):
            if not np.isnan(tr[i]) and not np.isnan(atr_50[i-1]):
                atr_50[i] = (atr_50[i-1] * 49 + tr[i]) / 50
    
    # Align HTF indicators to 6h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    prev_bull_power = 0
    prev_bear_power = 0
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]) or
            np.isnan(atr_14[i]) or np.isnan(atr_50[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volatility regime: low volatility (ATR14 < ATR50)
        low_vol_regime = atr_14[i] < atr_50[i]
        
        # Power momentum: rising bull power (less negative) or falling bear power (less positive)
        bull_power_rising = bull_power[i] > bull_power[i-1]
        bear_power_falling = bear_power[i] < bear_power[i-1]
        
        close_now = close_6h[i]
        ema_50_now = ema_50_aligned[i]
        ema_200_now = ema_200_aligned[i]
        bull_power_now = bull_power[i]
        bear_power_now = bear_power[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Bull Power > 0 AND rising AND 1d uptrend AND low volatility
            if (bull_power_now > 0 and bull_power_rising and 
                ema_50_now > ema_200_now and low_vol_regime):
                position = 1
                signals[i] = 0.25
            # Short conditions: Bear Power < 0 AND falling AND 1d downtrend AND low volatility
            elif (bear_power_now < 0 and bear_power_falling and 
                  ema_50_now < ema_200_now and low_vol_regime):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: power reverses or volatility regime shifts to high
            exit_long = (position == 1 and 
                        (bull_power_now <= 0 or not bull_power_rising or 
                         ema_50_now <= ema_200_now or not low_vol_regime))
            exit_short = (position == -1 and 
                         (bear_power_now >= 0 or not bear_power_falling or 
                          ema_50_now >= ema_200_now or not low_vol_regime))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        
        # Store current power for next iteration
        prev_bull_power = bull_power_now
        prev_bear_power = bear_power_now
    
    return signals