#!/usr/bin/env python3
"""
Hypothesis: 1h momentum strategy using 4h RSI trend filter and 1d volume regime filter.
Uses 1h RSI(14) for momentum signals, filtered by 4h RSI(50) > 50 for bull/bear regime,
and 1d volume > 1.2x 20-day average to ensure sufficient liquidity.
Designed to capture momentum bursts in trending markets while avoiding choppy, low-volume periods.
Target: 15-30 trades/year by requiring confluence of momentum, trend, and volume filters.
Works in bull markets (long bias) and bear markets (short bias) via 4h RSI regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1h RSI(14) for momentum signal ===
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi_1h = (100 - (100 / (1 + rs))).values
    
    # === 4h RSI(50) for trend regime filter ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    delta_4h = pd.Series(close_4h).diff()
    gain_4h = delta_4h.clip(lower=0)
    loss_4h = -delta_4h.clip(upper=0)
    avg_gain_4h = gain_4h.ewm(alpha=1/50, adjust=False, min_periods=50).mean()
    avg_loss_4h = loss_4h.ewm(alpha=1/50, adjust=False, min_periods=50).mean()
    rs_4h = avg_gain_4h / avg_loss_4h.replace(0, 1e-10)
    rsi_50_4h = (100 - (100 / (1 + rs_4h))).values
    rsi_50_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_50_4h)
    
    # === 1d volume regime filter ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    
    # Warmup: need enough data for RSI calculations
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(rsi_1h[i]) or np.isnan(rsi_50_4h_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume regime: today's volume > 1.2x 20-day average
        vol_today_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        vol_regime = vol_today_aligned[i] > vol_ma_20_1d_aligned[i] * 1.2
        
        # Skip if not in high-volume regime
        if not vol_regime:
            signals[i] = 0.0
            position = 0
            continue
        
        # Momentum signals
        rsi_overbought = rsi_1h[i] > 70
        rsi_oversold = rsi_1h[i] < 30
        
        # Trend filter from 4h RSI
        bull_regime = rsi_50_4h_aligned[i] > 50
        bear_regime = rsi_50_4h_aligned[i] < 50
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: oversold momentum in bull regime
            if rsi_oversold and bull_regime:
                signals[i] = 0.20
                position = 1
                continue
            # Short: overbought momentum in bear regime
            elif rsi_overbought and bear_regime:
                signals[i] = -0.20
                position = -1
                continue
        
        # Exit logic: exit on momentum reversal or regime change
        elif position == 1:
            # Exit long if overbought or regime turns bearish
            if rsi_overbought or not bull_regime:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short if oversold or regime turns bullish
            if rsi_oversold or not bear_regime:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI14_4hRSI50_1dVolumeRegime"
timeframe = "1h"
leverage = 1.0