#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1d ADX regime filter
# - Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# - Regime: ADX(14) > 25 = trending, < 20 = ranging (hysteresis)
# - In trending (ADX>25): Long if Bull Power > 0 and rising, Short if Bear Power < 0 and falling
# - In ranging (ADX<20): Long if Bear Power < -0.5*ATR and turning up, Short if Bull Power > 0.5*ATR and turning down
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Elder Ray measures bull/bear power relative to trend, effective in both bull and bear markets
# - ADX regime filter avoids whipsaws in ranging markets and captures trends when present
# - Works on BTC/ETH as it adapts to market regime

name = "6h_1d_elder_ray_adx_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for regime and Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # EMA(13) for Elder Ray
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power and Bear Power
    bull_power_1d = high_1d - ema_13_1d
    bear_power_1d = low_1d - ema_13_1d
    
    # ATR(14) for 1d
    tr_1d = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # ADX(14) for 1d
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    plus_dm = np.insert(plus_dm, 0, 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    tr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    plus_di_14_1d = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / tr_14_1d
    minus_di_14_1d = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / tr_14_1d
    dx_14_1d = 100 * np.abs(plus_di_14_1d - minus_di_14_1d) / (plus_di_14_1d + minus_di_14_1d)
    adx_14_1d = pd.Series(dx_14_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align all 1d indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    plus_di_14_aligned = align_htf_to_ltf(prices, df_1d, plus_di_14_1d)
    minus_di_14_aligned = align_htf_to_ltf(prices, df_1d, minus_di_14_1d)
    
    # Regime hysteresis: ADX > 25 = trending, ADX < 20 = ranging
    regime = np.zeros(n)  # 1=trending, -1=ranging, 0=transition
    adx_state = 0  # 0=unknown, 1=trending, -1=ranging
    for i in range(n):
        if not np.isnan(adx_14_aligned[i]):
            if adx_14_aligned[i] > 25:
                adx_state = 1
            elif adx_14_aligned[i] < 20:
                adx_state = -1
        regime[i] = adx_state
    
    for i in range(50, n):  # Start after 50-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(regime[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        bull_power = bull_power_aligned[i]
        bear_power = bear_power_aligned[i]
        atr_val = atr_14_aligned[i]
        reg = regime[i]
        
        # Previous values for momentum
        prev_bull = bull_power_aligned[i-1] if i > 0 else bull_power
        prev_bear = bear_power_aligned[i-1] if i > 0 else bear_power
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        if reg == 1:  # Trending regime
            # Long: Bull Power > 0 and rising
            if bull_power > 0 and bull_power > prev_bull:
                enter_long = True
            # Short: Bear Power < 0 and falling
            if bear_power < 0 and bear_power < prev_bear:
                enter_short = True
        elif reg == -1:  # Ranging regime
            # Long: Bear Power < -0.5*ATR and turning up
            if bear_power < -0.5 * atr_val and bear_power > prev_bear:
                enter_long = True
            # Short: Bull Power > 0.5*ATR and turning down
            if bull_power > 0.5 * atr_val and bull_power < prev_bull:
                enter_short = True
        
        # Exit conditions: opposite regime or power reversal
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if regime changes to ranging or Bull Power turns down
            exit_long = (reg == -1) or (bull_power < prev_bull)
        elif position == -1:
            # Exit short if regime changes to ranging or Bear Power turns up
            exit_short = (reg == -1) or (bear_power > prev_bear)
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals