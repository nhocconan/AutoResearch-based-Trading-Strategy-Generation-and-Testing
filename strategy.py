#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction, combined with RSI(14) for momentum and Choppiness Index for regime filtering. Enters long when KAMA slopes up, RSI > 50, and market is trending (CHOP < 38.2). Enters short when KAMA slopes down, RSI < 50, and market is trending. Uses discrete position sizing (0.25) to minimize fee churn and ATR-based stoploss for risk management. Designed to work in both bull and bear markets by adapting to trend regime and avoiding ranging markets where whipsaws occur.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need warmup for indicators
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for HTF trend context (optional filter)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on daily close
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, 1))
    change = np.insert(change, 0, 0)  # align length
    volatility = np.sum(np.abs(np.diff(close, 1)), axis=0)  # placeholder, will compute properly below
    
    # Proper KAMA calculation
    lookback = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    # Calculate efficiency ratio
    price_change = np.abs(np.diff(close, lookback))
    price_change = np.concatenate([np.full(lookback, np.nan), price_change])
    
    volatility_sum = np.zeros_like(close)
    for i in range(lookback, len(close)):
        volatility_sum[i] = np.sum(np.abs(np.diff(close[i-lookback:i+1, 0])))
    volatility_sum[:lookback] = np.nan
    
    er = np.where(volatility_sum > 0, price_change / volatility_sum, 0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate RSI(14)
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index (CHOP) for regime detection
    chop_window = 14
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    max_high = pd.Series(high).rolling(window=chop_window, min_periods=chop_window).max().values
    min_low = pd.Series(low).rolling(window=chop_window, min_periods=chop_window).min().values
    
    chop = np.zeros_like(close)
    for i in range(chop_window, len(close)):
        if atr[i] > 0 and not np.isnan(max_high[i]) and not np.isnan(min_low[i]):
            chop[i] = 100 * np.log10(np.sum(tr[i-chop_window+1:i+1]) / (np.log10(chop_window) * (max_high[i] - min_low[i]))) / np.log10(chop_window)
        else:
            chop[i] = np.nan
    chop[:chop_window] = np.nan
    
    # Calculate ATR(14) for stoploss
    atr_val = atr  # already calculated above
    
    # Calculate average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    base_size = 0.25
    atr_multiplier = 2.0  # ATR stoploss multiplier
    
    # Start after warmup (need 20 for volume, 34 for KAMA stability, 14 for ATR/RSI/CHOP)
    start_idx = max(20, 34, 14)
    
    for i in range(start_idx, n):
        # Get current values
        close_val = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        atr_val = atr_val[i]
        
        # Skip if any data not ready
        if (np.isnan(kama_val) or np.isnan(rsi_val) or np.isnan(chop_val) or 
            np.isnan(avg_vol) or np.isnan(atr_val)):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # KAMA slope for trend direction (using 2-period slope)
        kama_slope = kama_val - kama[i-1] if i > 0 else 0
        
        # Volume confirmation: current volume > 1.3x average volume
        volume_confirmed = vol > 1.3 * avg_vol
        
        # Regime filter: trending market (CHOP < 38.2)
        trending_regime = chop_val < 38.2
        
        # Long logic: KAMA trending up, RSI > 50, trending regime, volume confirmation
        long_condition = (kama_slope > 0) and (rsi_val > 50) and trending_regime and volume_confirmed
        # Short logic: KAMA trending down, RSI < 50, trending regime, volume confirmation
        short_condition = (kama_slope < 0) and (rsi_val < 50) and trending_regime and volume_confirmed
        
        # Exit logic: KAMA slope reversal or RSI extreme
        exit_long = (kama_slope <= 0) or (rsi_val < 40)
        exit_short = (kama_slope >= 0) or (rsi_val > 60)
        
        # ATR-based stoploss
        if position == 1:
            stop_price = entry_price - atr_multiplier * atr_val
            if close_val < stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:
            stop_price = entry_price + atr_multiplier * atr_val
            if close_val > stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val  # Enter at next bar open, approximate with close
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val  # Enter at next bar open, approximate with close
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "1d_KAMA_Trend_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0