#!/usr/bin/env python3
"""
1d_KAMA_Regime_Filter_DonchianExit
Hypothesis: On 1d timeframe, use KAMA(10,2,30) to determine trend direction, 
filtered by choppiness index (CHOP > 61.8 = range, < 38.2 = trend). 
Enter long when KAMA up AND trending regime, short when KAMA down AND trending regime.
Exit on Donchian(20) breakout in opposite direction. 
ATR(14) stoploss (2.5x) and discrete sizing (0.25).
Uses 1w HTF for higher-timeframe trend alignment to avoid counter-trend trades.
Target: 30-100 total trades over 4 years (7-25/year) to avoid fee drag.
Works in both bull (trend following) and bear (range mean-reversion via regime filter).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 150:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for trend alignment)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # === 1d KAMA (Efficiency Ratio = 10, Fast=2, Slow=30) ===
    close = prices['close'].values
    # Calculate Efficiency Ratio
    change = np.abs(np.diff(close, n=10))  # 10-period net change
    vol = np.sum(np.abs(np.diff(close)), axis=0)  # 10-period volatility (sum of abs changes)
    # Avoid division by zero
    er = np.where(vol != 0, change / vol, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Initialize KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start after 10 periods
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === 1d Choppiness Index (CHOP) ===
    high = prices['high'].values
    low = prices['low'].values
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Set first TR to high-low (no previous close)
    tr[0] = high[0] - low[0]
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # CHOP = 100 * log10(tr_sum / (hh - ll)) / log10(14)
    # Avoid division by zero and log of zero
    range_hl = hh - ll
    chop = np.where((range_hl > 0) & (tr_sum > 0), 
                    100 * np.log10(tr_sum / range_hl) / np.log10(14), 
                    50)  # Default to neutral when invalid
    
    # === 1d Donchian(20) for exit ===
    dc_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    dc_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1d ATR(14) for stoploss ===
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === 1w EMA34 for HTF trend filter ===
    df_1w_close = df_1w['close'].values
    ema_34_1w = pd.Series(df_1w_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(kama[i]) or np.isnan(chop[i]) or np.isnan(dc_high[i]) or 
            np.isnan(dc_low[i]) or np.isnan(atr[i]) or np.isnan(ema_34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        kama_now = kama[i]
        kama_prev = kama[i-1]
        chop_now = chop[i]
        dc_high_val = dc_high[i]
        dc_low_val = dc_low[i]
        htf_trend = ema_34_1w_aligned[i]
        
        # KAMA direction: rising if current > previous
        kama_up = kama_now > kama_prev
        kama_down = kama_now < kama_prev
        
        # Regime filter: trending if CHOP < 38.2, ranging if CHOP > 61.8
        trending_regime = chop_now < 38.2
        ranging_regime = chop_now > 61.8
        
        if position == 0:
            # Enter long: KAMA up AND trending regime AND price above HTF EMA (bullish bias)
            # Enter short: KAMA down AND trending regime AND price below HTF EMA (bearish bias)
            long_condition = kama_up and trending_regime and (price > htf_trend)
            short_condition = kama_down and trending_regime and (price < htf_trend)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit conditions for long
            exit = False
            # Stoploss: 2.5x ATR
            if price < entry_price - 2.5 * atr[i]:
                exit = True
            # Donchian breakout: price breaks below 20-period low
            elif price < dc_low_val:
                exit = True
            # Regime change to ranging: avoid whipsaw in sideways markets
            elif ranging_regime:
                exit = True
            # HTF trend reversal: price crosses below weekly EMA
            elif price < htf_trend:
                exit = True
            
            if exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions for short
            exit = False
            # Stoploss: 2.5x ATR
            if price > entry_price + 2.5 * atr[i]:
                exit = True
            # Donchian breakout: price breaks above 20-period high
            elif price > dc_high_val:
                exit = True
            # Regime change to ranging: avoid whipsaw in sideways markets
            elif ranging_regime:
                exit = True
            # HTF trend reversal: price crosses above weekly EMA
            elif price > htf_trend:
                exit = True
            
            if exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Regime_Filter_DonchianExit"
timeframe = "1d"
leverage = 1.0