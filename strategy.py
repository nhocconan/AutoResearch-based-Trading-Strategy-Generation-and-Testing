#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR-based volatility filter and 1w trend filter
# Uses 4h primary timeframe with 1d HTF for volatility regime (low ATR = range, high ATR = trend)
# and 1w HTF for trend direction (price above/below 50-period EMA). Only takes breakouts
# in the direction of the weekly trend when volatility is elevated (avoids false breakouts in low-volatility regimes).
# Volatility filter ensures we only trade when there is sufficient momentum to sustain a breakout.
# Weekly trend filter ensures we trade with the higher timeframe momentum, improving win rate in both bull and bear markets.
# Target: 75-200 trades over 4 years (19-50/year) to balance statistical significance and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 4h data (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # === 1d data (HTF for ATR-based volatility filter) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1w data (HTF for trend filter) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # === 1d ATR (20-period) for volatility regime ===
    tr_1d = np.maximum(high_1d - low_1d, np.absolute(high_1d - np.roll(close_1d, 1)), np.absolute(low_1d - np.roll(close_1d, 1)))
    tr_1d[0] = high_1d[0] - low_1d[0]  # first bar
    atr_1d = pd.Series(tr_1d).rolling(window=20, min_periods=20).mean().values
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values  # 20-period MA of ATR
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
    
    # === 1d ATR (current) for volatility ratio ===
    atr_1d_current = pd.Series(tr_1d).rolling(window=20, min_periods=20).mean().values
    atr_1d_current_aligned = align_htf_to_ltf(prices, df_1d, atr_1d_current)
    
    # === 1w EMA(50) for trend filter ===
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === 4h Donchian channels (20-period) ===
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe (wait for 4h bar close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_regime_aligned[i]) or
            np.isnan(atr_1d_current_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        vol_regime = vol_regime_aligned[i]
        atr_current = atr_1d_current_aligned[i]
        weekly_ema = ema_50_1w_aligned[i]
        
        # Volatility filter: only trade when current ATR > 1.5 * 20-period MA of ATR (elevated volatility)
        vol_filter = atr_current > (1.5 * vol_regime)
        
        # Weekly trend filter: price above/below weekly EMA50
        weekly_uptrend = price > weekly_ema
        weekly_downtrend = price < weekly_ema
        
        # === STOPLOSS LOGIC (ATR-based) ===
        if position == 1:  # Long position
            atr_4h = np.maximum(high_4h - low_4h, np.absolute(high_4h - np.roll(close_4h, 1)), np.absolute(low_4h - np.roll(close_4h, 1)))
            atr_4h[0] = high_4h[0] - low_4h[0]
            atr_ma_4h = pd.Series(atr_4h).rolling(window=20, min_periods=20).mean().values
            atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_ma_4h)
            atr_val = atr_4h_aligned[i]
            if price < entry_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            atr_4h = np.maximum(high_4h - low_4h, np.absolute(high_4h - np.roll(close_4h, 1)), np.absolute(low_4h - np.roll(close_4h, 1)))
            atr_4h[0] = high_4h[0] - low_4h[0]
            atr_ma_4h = pd.Series(atr_4h).rolling(window=20, min_periods=20).mean().values
            atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_ma_4h)
            atr_val = atr_4h_aligned[i]
            if price > entry_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price reaches Donchian low or weekly trend turns down
            if price <= lower or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price reaches Donchian high or weekly trend turns up
            if price >= upper or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require elevated volatility and weekly trend alignment
            if vol_filter:
                # Go long when price breaks above Donchian high and weekly trend is up
                if price > upper and weekly_uptrend:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                # Go short when price breaks below Donchian low and weekly trend is down
                elif price < lower and weekly_downtrend:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_VolatilityRegime_WeeklyTrend"
timeframe = "4h"
leverage = 1.0