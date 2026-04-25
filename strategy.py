#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dEMA34_RegimeFilter_v1
Hypothesis: Trade Camarilla R1/S1 breakouts on 4h with 1d EMA34 trend filter and volatility regime filter. Uses choppiness index (CHOP) to avoid whipsaws in ranging markets. Only takes breakouts aligned with 1d trend when market is trending (CHOP < 38.2) or mean-reverts at extremes when choppy (CHOP > 61.8). Volume confirmation (2.0x average) ensures institutional participation. Designed for 4h timeframe with tight entries (~20-30/year) to minimize fee drag while capturing strong directional moves and avoiding false breakouts in chop.
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
    
    # Get 1d data for HTF trend filter (EMA34) and Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels using previous 1d bar's OHLC
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d) / 12.0
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d) / 12.0
    
    # Align Camarilla levels to 4h timeframe (previous 1d bar's levels available)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: 2.0x 20-bar average volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    # Choppiness Index regime filter (14-period)
    def choppiness_index(high, low, close, period=14):
        atr = np.zeros(len(close))
        tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
        atr[period:] = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
        atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
        hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
        ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
        chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(period)
        return chop
    
    chop = choppiness_index(high, low, close, 14)
    chop_trending = chop < 38.2   # Trending regime
    chop_choppy = chop > 61.8     # Choppy/ranging regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34(1d) and volume MA(20)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend
        htf_1d_bullish = close[i] > ema_34_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Look for Camarilla breakouts with volume confirmation
            long_breakout = (high[i] > camarilla_r1_aligned[i]) and volume_spike[i]
            short_breakout = (low[i] < camarilla_s1_aligned[i]) and volume_spike[i]
            
            # Regime-based entry logic
            if chop_trending[i]:
                # Trending market: only trade breakouts in direction of 1d trend
                if long_breakout and htf_1d_bullish:
                    signals[i] = 0.25
                    position = 1
                elif short_breakout and htf_1d_bearish:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif chop_choppy[i]:
                # Choppy market: mean reversion at extremes
                if long_breakout and not htf_1d_bullish:  # Fade bullish breakout in chop
                    signals[i] = -0.25
                    position = -1
                elif short_breakout and not htf_1d_bearish:  # Fade bearish breakout in chop
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            else:
                # Neutral regime: no trading
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price returns to Camarilla H3/L3 level or regime becomes unfavorable
            camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 6.0
            camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
            exit_signal = (low[i] < camarilla_h3_aligned[i]) or \
                         (chop_choppy[i] and not htf_1d_bullish) or \
                         (chop_trending[i] and not htf_1d_bullish)
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price returns to Camarilla L3/H3 level or regime becomes unfavorable
            camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 6.0
            camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
            exit_signal = (high[i] > camarilla_l3_aligned[i]) or \
                         (chop_choppy[i] and htf_1d_bearish) or \
                         (chop_trending[i] and htf_1d_bearish)
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_RegimeFilter_v1"
timeframe = "4h"
leverage = 1.0