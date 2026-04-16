#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w trend filter and volume confirmation.
# Long when price > Alligator Jaw (13-period SMMA) AND 1w close > 1w open (bullish weekly candle) AND volume > 1.5x 20-day average.
# Short when price < Alligator Jaw AND 1w close < 1w open (bearish weekly candle) AND volume > 1.5x 20-day average.
# Exit on ATR-based stoploss (2*ATR from entry) or opposite Alligator Jaw crossover.
# Uses discrete position size 0.25. Designed to capture trending moves with volume confirmation in both bull and bear markets.
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Indicators: Williams Alligator (Smoothed Moving Averages) ===
    # Jaw: 13-period SMMA of median price, smoothed 8 periods
    median_price = (high + low) / 2.0
    smma_jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean()
    smma_jaw = smma_jaw.ewm(alpha=1/8, adjust=False, min_periods=8).mean().values
    
    # === 1w Indicators: Weekly trend and volume confirmation ===
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    weekly_open = df_1w['open'].values
    weekly_bullish = weekly_close > weekly_open  # Bullish weekly candle
    weekly_bearish = weekly_close < weekly_open  # Bearish weekly candle
    
    # Volume confirmation: 1d volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    # === 1d ATR for stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Align HTF indicators to LTF
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    smma_jaw_aligned = align_htf_to_ltf(prices, df_1w, smma_jaw)  # Align Jaw to 1d
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(smma_jaw_aligned[i]) or np.isnan(weekly_bullish_aligned[i]) or
            np.isnan(weekly_bearish_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(atr[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        jaw = smma_jaw_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        wk_bullish = weekly_bullish_aligned[i] > 0.5
        wk_bearish = weekly_bearish_aligned[i] > 0.5
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below Alligator Jaw
            if price < jaw:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above Alligator Jaw
            if price > jaw:
                exit_signal = True
            # ATR-based stoploss: 2*ATR above entry
            elif price > entry_price + 2.0 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price > Jaw AND weekly bullish AND volume spike
            if price > jaw and wk_bullish and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price < Jaw AND weekly bearish AND volume spike
            elif price < jaw and wk_bearish and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "1d_WilliamsAlligator_1wTrend_VolumeSpike_V1"
timeframe = "1d"
leverage = 1.0