#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d Regime Filter
# - Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# - Long: Bull Power > 0 AND Bear Power < 0 AND price > 1d EMA(50) AND volume > 1.2x 20-period avg
# - Short: Bear Power > 0 AND Bull Power < 0 AND price < 1d EMA(50) AND volume > 1.2x 20-period avg
# - Exit: Power values converge (|Bull Power| < 0.1 * ATR AND |Bear Power| < 0.1 * ATR)
# - Uses 6h timeframe for lower frequency (target: 12-30 trades/year)
# - Elder Ray measures bull/bear strength relative to EMA, effective in both trending and ranging markets
# - 1d EMA filter ensures alignment with higher timeframe trend
# - Volume confirmation reduces false signals
# - Discrete position sizing (0.25) minimizes fee churn

name = "6h_1d_elder_ray_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for EMA trend filter (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 6h EMA(13) for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Pre-compute ATR for exit condition
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute 6h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_13[i]) or np.isnan(atr[i]) or 
            np.isnan(volume_sma_20[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        high_price = high[i]
        low_price = low[i]
        volume_current = volume[i]
        
        # Elder Ray components
        bull_power = high_price - ema_13[i]
        bear_power = ema_13[i] - low_price
        
        # Volume confirmation: current volume > 1.2x 20-period average
        vol_confirm = volume_current > 1.2 * volume_sma_20[i]
        
        # 1d EMA trend bias
        ema_bias_long = close_price > ema_50_1d_aligned[i]
        ema_bias_short = close_price < ema_50_1d_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: strong bull power, weak bear power, long bias, volume confirmation
        if bull_power > 0 and bear_power < 0 and ema_bias_long and vol_confirm:
            enter_long = True
        
        # Short: strong bear power, weak bull power, short bias, volume confirmation
        if bear_power > 0 and bull_power < 0 and ema_bias_short and vol_confirm:
            enter_short = True
        
        # Exit conditions: power values converge (market losing directional strength)
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long when bull power weakens and bear power strengthens
            exit_long = (bull_power < 0.1 * atr[i]) and (bear_power > -0.1 * atr[i])
        elif position == -1:
            # Exit short when bear power weakens and bull power strengthens
            exit_short = (bear_power < 0.1 * atr[i]) and (bull_power > -0.1 * atr[i])
        
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