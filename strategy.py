#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h EMA crossover with 1d ADX regime filter and volume spike confirmation
# - Primary: 6h EMA(9) crossing above/below EMA(21) for trend changes
# - HTF: 1d ADX > 20 to ensure we're in a trending market (avoid whipsaws in ranging)
# - Volume confirmation: 6h volume > 1.8x 20-period MA to avoid false breakouts
# - Long: EMA9 > EMA21 + ADX > 20 + volume spike
# - Short: EMA9 < EMA21 + ADX > 20 + volume spike
# - Exit: Opposite EMA crossover or ADX drops below 15 (trend weakening)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: ADX filters ranging markets, EMA captures trends, volume confirms momentum
# - Target: 60-120 trades over 4 years (15-30/year) to stay within fee drag limits

name = "6h_1d_ema_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for ADX
        return np.zeros(n)
    
    # Pre-compute 6h data
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 6h EMA(9) and EMA(21)
    ema9_6h = pd.Series(close_6h).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema21_6h = pd.Series(close_6h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 6h volume moving average (20-period) for volume confirmation
    volume_ma_20_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ADX (1d) for trend strength
    # True Range
    tr1 = np.abs(np.roll(high_1d, 1) - np.roll(low_1d, 1))
    tr2 = np.abs(np.roll(high_1d, 1) - np.roll(close_1d, 1))
    tr3 = np.abs(np.roll(low_1d, 1) - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = np.roll(high_1d, 1) - high_1d
    down_move = low_1d - np.roll(low_1d, 1)
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values (using Wilder's smoothing = EMA with alpha=1/period)
    period = 14
    alpha = 1.0 / period
    
    atr = pd.Series(tr).ewm(alpha=alpha, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean().values / atr
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=alpha, adjust=False).mean().values
    
    # Align all HTF indicators to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(ema9_6h[i]) or np.isnan(ema21_6h[i]) or
            np.isnan(volume_ma_20_6h[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.8x 20-period MA
        volume_confirm = volume_6h[i] > 1.8 * volume_ma_20_6h[i]
        
        # ADX trend filter: ADX > 20 indicates trending market
        trend_confirm = adx_aligned[i] > 20.0
        
        # EMA crossover conditions
        ema_bullish = ema9_6h[i] > ema21_6h[i]
        ema_bearish = ema9_6h[i] < ema21_6h[i]
        
        # Exit conditions: Opposite EMA crossover or ADX drops below 15 (trend weakening)
        exit_long = ema_bearish or (adx_aligned[i] < 15.0)
        exit_short = ema_bullish or (adx_aligned[i] < 15.0)
        
        if position == 0:  # Flat - look for new entries
            # Long entry: EMA bullish crossover + volume confirmation + trend confirmation
            if ema_bullish and volume_confirm and trend_confirm:
                position = 1
                signals[i] = 0.25
            # Short entry: EMA bearish crossover + volume confirmation + trend confirmation
            elif ema_bearish and volume_confirm and trend_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            if position == 1:  # Long position
                if exit_long:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if exit_short:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals