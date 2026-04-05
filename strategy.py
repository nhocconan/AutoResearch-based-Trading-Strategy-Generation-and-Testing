#!/usr/bin/env python3
"""
Experiment #8167: 6-hour Ichimoku Cloud with 1-day trend filter and volume confirmation.
Hypothesis: In both bull and bear markets, price breaking above/below the Kumo (cloud) on 6h 
with Tenkan-Kijun cross aligned to 1d trend (price above/below 1d SMA50) and volume >1.5x 
20-period MA captures sustained moves while avoiding whipsaw. The Ichimoku cloud provides 
dynamic support/resistance that adapts to volatility, reducing false signals during 
consolidation.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8167_6h_ichimoku1d_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_PERIOD = 9
KIJUN_PERIOD = 26
SENKOU_B_PERIOD = 52
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
SMA_PERIOD = 50
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
ATR_TARGET_MULTIPLIER = 3.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d SMA
    close_1d = df_1d['close'].values
    sma_1d = pd.Series(close_1d).rolling(window=SMA_PERIOD, min_periods=SMA_PERIOD).mean().values
    
    # Price relative to SMA: above = bullish bias, below = bearish bias
    price_vs_sma = np.where(close_1d > sma_1d, 1, -1)  # 1=bullish, -1=bearish
    price_vs_sma_aligned = align_htf_to_ltf(prices, df_1d, price_vs_sma)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() + 
                  pd.Series(low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    tenkan_sen = tenkan_sen.values
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() + 
                 pd.Series(low).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    kijun_sen = kijun_sen.values
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    # Shift will be handled by alignment - we calculate current values
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b = (pd.Series(high).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).max() + 
                pd.Series(low).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).min()) / 2
    senkou_b = senkou_b.values
    
    # Current Kumo (cloud) boundaries - use Senkou A/B from 26 periods ago
    # For simplicity, we use current Senkou A/B as cloud edges (simplified but functional)
    kumo_top = np.maximum(senkou_a, senkou_b)  # Upper cloud boundary
    kumo_bottom = np.minimum(senkou_a, senkou_b)  # Lower cloud boundary
    
    # Tenkan-Kijun cross
    tk_cross = np.where(tenkan_sen > kijun_sen, 1, -1)  # 1=bullish cross, -1=bearish cross
    
    # Price above/below cloud
    price_above_kumo = np.where(close > kumo_top, 1, 0)
    price_below_kumo = np.where(close < kumo_bottom, 1, 0)
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    
    # Start from warmup period
    start = max(TENKAN_PERIOD, KIJUN_PERIOD, SENKOU_B_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD, SMA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_vs_sma_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss or target
        if position == 1:  # long position
            if close[i] <= stop_price or close[i] >= target_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price or close[i] <= target_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine market bias from 1d SMA
        bull_bias = price_vs_sma_aligned[i] == 1   # 1d close above SMA50
        bear_bias = price_vs_sma_aligned[i] == -1  # 1d close below SMA50
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Ichimoku signals
        # Bullish: price above cloud + TK bullish cross
        ichimoku_bullish = (price_above_kumo[i] == 1 and tk_cross[i] == 1)
        # Bearish: price below cloud + TK bearish cross
        ichimoku_bearish = (price_below_kumo[i] == 1 and tk_cross[i] == -1)
        
        # Entry conditions
        long_entry = bull_bias and ichimoku_bullish and volume_confirmed
        short_entry = bear_bias and ichimoku_bearish and volume_confirmed
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price + (ATR_TARGET_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price - (ATR_TARGET_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals