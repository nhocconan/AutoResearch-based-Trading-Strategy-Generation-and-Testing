#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dEMA34_Trend_ChopFilter_v2
Hypothesis: 4-hour Camarilla R1/S1 level breakout with daily EMA34 trend filter and choppiness regime (CHOP > 61.8 = range) for mean reversion entries. Uses discrete position sizing (0.30) and ATR-based stoploss (2.0x) for risk management. Designed for moderate trade frequency (target 20-50/year) to minimize fee drag while capturing reversals in ranging markets and trends in trending markets. The daily EMA34 provides strong trend filtering that works across regimes, and choppiness filter avoids false breakouts in strong trends. Focuses on BTC and ETH as primary targets.
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
    
    # Get daily data for HTF filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA(34) on daily for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for stoploss on 4h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate choppiness index on daily for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / (n * (max(high) - min(low)))) / log10(n)
    # We'll use a simplified version: CHOP > 61.8 = range, CHOP < 38.2 = trend
    # For daily, calculate TR sum over 14 days and price range over 14 days
    tr_1d1 = df_1d['high'][1:] - df_1d['low'][1:]
    tr_1d2 = np.abs(df_1d['high'][1:] - df_1d['close'][:-1])
    tr_1d3 = np.abs(df_1d['low'][1:] - df_1d['close'][:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr_1d1, np.maximum(tr_1d2, tr_1d3))])
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    price_range_14 = max_high_14 - min_low_14
    chop_1d = 100 * np.log10(atr_14_1d / (14 * price_range_14)) / np.log10(14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate Camarilla levels from previous daily bar
    # Camarilla: R1 = C + ((H-L)*1.1/12), S1 = C - ((H-L)*1.1/12)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla R1 and S1
    camarilla_r1 = prev_close + ((prev_high - prev_low) * 1.1 / 12)
    camarilla_s1 = prev_close - ((prev_high - prev_low) * 1.1 / 12)
    
    # Align Camarilla levels to 4h (use previous day's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of daily EMA(34), ATR, chop
    start_idx = max(34, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(chop_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        close_val = close[i]
        trend_1d_up = close_val > ema_34_1d_aligned[i]   # Daily uptrend
        trend_1d_down = close_val < ema_34_1d_aligned[i]  # Daily downtrend
        chop_val = chop_1d_aligned[i]
        is_range = chop_val > 61.8  # Chop > 61.8 = ranging market
        is_trend = chop_val < 38.2  # Chop < 38.2 = trending market
        
        if position == 0:
            # In ranging market: mean reversion at Camarilla S1/R1
            # Long: price crosses above Camarilla S1 AND ranging market
            long_signal = (close_val > camarilla_s1_aligned[i]) and is_range
            
            # Short: price crosses below Camarilla R1 AND ranging market
            short_signal = (close_val < camarilla_r1_aligned[i]) and is_range
            
            # In trending market: breakout with trend
            # Long: price breaks above Camarilla R1 AND daily uptrend AND trending market
            long_signal_trend = (close_val > camarilla_r1_aligned[i]) and trend_1d_up and is_trend
            
            # Short: price breaks below Camarilla S1 AND daily downtrend AND trending market
            short_signal_trend = (close_val < camarilla_s1_aligned[i]) and trend_1d_down and is_trend
            
            if long_signal or long_signal_trend:
                signals[i] = 0.30
                position = 1
                entry_price = close_val
            elif short_signal or short_signal_trend:
                signals[i] = -0.30
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.30
            # Exit: trend flips down OR price hits ATR stoploss OR opposite Camarilla level touched
            if (not trend_1d_up) or (close_val < entry_price - 2.0 * atr[i]) or (close_val < camarilla_s1_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.30
            # Exit: trend flips up OR price hits ATR stoploss OR opposite Camarilla level touched
            if (not trend_1d_down) or (close_val > entry_price + 2.0 * atr[i]) or (close_val > camarilla_r1_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_Trend_ChopFilter_v2"
timeframe = "4h"
leverage = 1.0