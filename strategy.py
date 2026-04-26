#!/usr/bin/env python3
"""
6h_ADX_TrendStrength_KeltnerBreakout_v1
Hypothesis: 6h strategy using ADX > 25 to confirm trending markets + Keltner Channel breakouts with volume confirmation.
- Long when price breaks above Keltner Upper Band AND ADX > 25 AND volume spike
- Short when price breaks below Keltner Lower Band AND ADX > 25 AND volume spike
- Keltner Channels adapt to volatility (ATR-based), reducing false breakouts in ranging markets
- ADX filter ensures we only trade strong trends, avoiding whipsaws in sideways markets
- Volume spike (2.0x 20-period average) confirms institutional participation
- Designed for medium frequency with proven edge on BTC/ETH from trend-following + volatility breakout logic
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR for Keltner Channels (14-period)
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.roll(close, 1))).values
    tr3 = pd.Series(np.abs(low - np.roll(close, 1))).values
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate EMA for Keltner Channel middle line (20-period)
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channels: EMA20 ± (ATR * 2.0)
    keltner_upper = ema20 + (atr * 2.0)
    keltner_lower = ema20 - (atr * 2.0)
    
    # Calculate ADX (14-period) for trend strength filter
    # +DM and -DM calculation
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = np.nan
    down_move[0] = np.nan
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth the DM values
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Calculate +DI and -DI
    plus_di = 100 * (plus_dm_smooth / tr_sum)
    minus_di = 100 * (minus_dm_smooth / tr_sum)
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume spike (20-period volume average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)  # Volume at least 2.0x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for ADX: 14+14+14-2 for smoothing)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx[i]) or 
            np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # ADX > 25 indicates strong trend
        strong_trend = adx[i] > 25
        
        if position == 0:
            # Long: Price breaks above Keltner Upper Band AND strong trend AND volume spike
            if close[i] > keltner_upper[i] and strong_trend and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Keltner Lower Band AND strong trend AND volume spike
            elif close[i] < keltner_lower[i] and strong_trend and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls back below EMA20 (middle of Keltner) OR trend weakens
            if close[i] < ema20[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises back above EMA20 OR trend weakens
            if close[i] > ema20[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ADX_TrendStrength_KeltnerBreakout_v1"
timeframe = "6h"
leverage = 1.0