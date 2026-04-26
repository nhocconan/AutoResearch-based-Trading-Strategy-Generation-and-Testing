#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike_ATRStop
Hypothesis: On 4h timeframe, Donchian channel (20) breakouts with 1d EMA34 trend filter and volume spike (>2x 20-bar average) capture strong momentum moves in both bull and bear markets. ATR-based stoploss (2.5x ATR) limits drawdown. Targets 20-40 trades/year to minimize fee drag while maintaining edge via trend and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA34 on 1d for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # ATR(14) for volatility and stoploss
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.roll(close, 1))).values
    tr3 = pd.Series(np.abs(low - np.roll(close, 1))).values
    tr2[0] = tr1[0]  # first bar
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period) for volume spike confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(40, 34, 20, 20, 14)  # 1d lookback, EMA34, Donchian, volume MA, ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(atr[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        ema_34_val = ema_34_aligned[i]
        donchian_high_val = donchian_high[i]
        donchian_low_val = donchian_low[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        atr_val = atr[i]
        
        # Volume confirmation: current volume > 2x 20-period average
        volume_confirmed = vol_val > 2.0 * vol_ma_val
        
        if position == 0:
            # Long: price breaks above Donchian high with uptrend (close > EMA34) and volume confirmation
            long_signal = (high_val > donchian_high_val) and (close_val > ema_34_val) and volume_confirmed
            # Short: price breaks below Donchian low with downtrend (close < EMA34) and volume confirmation
            short_signal = (low_val < donchian_low_val) and (close_val < ema_34_val) and volume_confirmed
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                atr_at_entry = atr_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                atr_at_entry = atr_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. ATR-based stoploss: price closes below entry - 2.5 * ATR_at_entry
            if close_val < entry_price - 2.5 * atr_at_entry:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                atr_at_entry = 0.0
            # 2. Trend reversal: close crosses below EMA34
            elif close_val < ema_34_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                atr_at_entry = 0.0
            # 3. Donchian reversal: price breaks below Donchian low (exit long)
            elif low_val < donchian_low_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                atr_at_entry = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. ATR-based stoploss: price closes above entry + 2.5 * ATR_at_entry
            if close_val > entry_price + 2.5 * atr_at_entry:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                atr_at_entry = 0.0
            # 2. Trend reversal: close crosses above EMA34
            elif close_val > ema_34_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                atr_at_entry = 0.0
            # 3. Donchian reversal: price breaks above Donchian high (exit short)
            elif high_val > donchian_high_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                atr_at_entry = 0.0
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0