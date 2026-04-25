#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA50_RegimeADX
Hypothesis: Camarilla R1/S1 breakouts on 4h with 1d EMA50 trend filter and ADX regime filter (ADX>25 for trending, ADX<20 for ranging). Only trades breakouts in direction of 1d trend with volume confirmation (>1.5x 20-bar avg). Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (<30/year) to work in both bull/bear markets via trend alignment and regime filtering.
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
    
    # Get 1d data for HTF trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA50 on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate ADX on 1d data for regime filter
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(tr)
        plus_di = np.zeros_like(tr)
        minus_di = np.zeros_like(tr)
        dx = np.zeros_like(tr)
        adx = np.zeros_like(tr)
        
        atr[period] = np.mean(tr[1:period+1])
        plus_dm_smooth = np.mean(plus_dm[1:period+1])
        minus_dm_smooth = np.mean(minus_dm[1:period+1])
        
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth = (plus_dm_smooth * (period-1) + plus_dm[i]) / period
            minus_dm_smooth = (minus_dm_smooth * (period-1) + minus_dm[i]) / period
            plus_di[i] = 100 * plus_dm_smooth / atr[i] if atr[i] != 0 else 0
            minus_di[i] = 100 * minus_dm_smooth / atr[i] if atr[i] != 0 else 0
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) if (plus_di[i] + minus_di[i]) != 0 else 0
        
        # ADX is smoothed DX
        adx[2*period] = np.mean(dx[period+1:2*period+1])
        for i in range(2*period+1, len(tr)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Calculate Camarilla levels on 1d data (based on previous bar's OHLC)
    camarilla_r1_1d = close_1d + ((high_1d - low_1d) * 1.1 / 12)
    camarilla_s1_1d = close_1d - ((high_1d - low_1d) * 1.1 / 12)
    camarilla_h4_1d = close_1d + ((high_1d - low_1d) * 1.1 / 2)
    camarilla_l4_1d = close_1d - ((high_1d - low_1d) * 1.1 / 2)
    
    # Align HTF indicators to 4h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d, additional_delay_bars=1)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d, additional_delay_bars=1)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d, additional_delay_bars=1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d, additional_delay_bars=1)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4_1d, additional_delay_bars=1)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4_1d, additional_delay_bars=1)
    
    # Volume confirmation: 1.5x 20-bar average volume (moderate filter)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50) and ADX (~28+14=42) and volume MA (20)
    start_idx = max(50, 42, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or 
            np.isnan(adx_aligned[i]) or
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or
            np.isnan(camarilla_l4_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Determine regime: trending (ADX>25) or ranging (ADX<20)
            is_trending = adx_aligned[i] > 25
            is_ranging = adx_aligned[i] < 20
            
            # Look for breakout signals with trend filter and volume spike
            # Long: price breaks above R1 in uptrend (close > EMA50) with volume spike
            # Short: price breaks below S1 in downtrend (close < EMA50) with volume spike
            long_signal = (close[i] > camarilla_r1_aligned[i]) and (close[i] > ema50_aligned[i]) and volume_spike[i]
            short_signal = (close[i] < camarilla_s1_aligned[i]) and (close[i] < ema50_aligned[i]) and volume_spike[i]
            
            # In trending regime, only trade with trend; in ranging regime, trade both directions
            if is_trending:
                if long_signal and (close[i] > ema50_aligned[i]):  # Uptrend confirmation
                    signals[i] = 0.25
                    position = 1
                elif short_signal and (close[i] < ema50_aligned[i]):  # Downtrend confirmation
                    signals[i] = -0.25
                    position = -1
            else:  # ranging regime or transition
                if long_signal:
                    signals[i] = 0.25
                    position = 1
                elif short_signal:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price moves back below Camarilla H4 (take profit at resistance)
            exit_signal = close[i] < camarilla_h4_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above Camarilla L4 (take profit at support)
            exit_signal = close[i] > camarilla_l4_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA50_RegimeADX"
timeframe = "4h"
leverage = 1.0