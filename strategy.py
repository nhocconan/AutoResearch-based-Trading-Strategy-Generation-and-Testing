#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ATR-based breakout with 1d volume spike filter and weekly trend confirmation
# Long when price breaks above ATR(14) upper band AND 1d volume > 2.0x 20-period median AND weekly close > weekly EMA50
# Short when price breaks below ATR(14) lower band AND 1d volume > 2.0x 20-period median AND weekly close < weekly EMA50
# Exit when price retraces to 6h EMA20 (dynamic stop/re-entry zone)
# Uses discrete position size 0.25 to limit fee drag. Target: 50-150 total trades over 4 years.
# Combines volatility breakout with volume confirmation and weekly trend filter for robustness in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: Volume median (20-period) ===
    volume_1d = df_1d['volume'].values
    vol_median_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).median().values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # === Weekly Indicators: EMA50 trend filter ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === 6h Indicators: ATR(14) bands and EMA20 ===
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ATR bands: upper = close + 2.0*atr, lower = close - 2.0*atr
    atr_upper = close + 2.0 * atr_14
    atr_lower = close - 2.0 * atr_14
    
    # 6h EMA20 for exit
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align all indicators to primary timeframe
    vol_median_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_median_20_1d)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    atr_upper_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), atr_upper)  # dummy df for alignment
    atr_lower_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), atr_lower)
    ema_20_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), ema_20)
    
    # Get aligned 1d volume for volume spike check
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 14, 50)  # 6h EMA20, ATR14, weekly EMA50
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_upper_aligned[i]) or np.isnan(atr_lower_aligned[i]) or 
            np.isnan(ema_20_aligned[i]) or np.isnan(vol_median_20_1d_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current values (aligned)
        price = close[i]
        atr_upper_val = atr_upper_aligned[i]
        atr_lower_val = atr_lower_aligned[i]
        ema_20_val = ema_20_aligned[i]
        vol_median = vol_median_20_1d_aligned[i]
        weekly_ema = ema_50_1w_aligned[i]
        vol_1d = volume_1d_aligned[i]
        
        # Volume filter: current 1d volume > 2.0x 20-period 1d volume median
        vol_threshold = vol_median * 2.0
        vol_confirm = vol_1d > vol_threshold
        
        # Weekly trend filter
        weekly_trend_up = weekly_ema < price  # Simplified: price above weekly EMA = uptrend
        weekly_trend_down = weekly_ema > price  # price below weekly EMA = downtrend
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit when price retraces to 6h EMA20
            if price <= ema_20_val:
                exit_signal = True
        elif position == -1:  # short position
            # Exit when price retraces to 6h EMA20
            if price >= ema_20_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG CONDITIONS
            # Price breaks above ATR upper band AND volume confirmation AND weekly uptrend
            if price > atr_upper_val and vol_confirm and weekly_trend_up:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT CONDITIONS
            # Price breaks below ATR lower band AND volume confirmation AND weekly downtrend
            elif price < atr_lower_val and vol_confirm and weekly_trend_down:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "6h_ATRBreakout_1dVolumeSpike_1wTrend_v1"
timeframe = "6h"
leverage = 1.0