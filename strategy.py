#!/usr/bin/env python3
"""
Experiment #8475: 6h Ichimoku Cloud with 1d Trend Filter + Volume Confirmation
Hypothesis: Ichimoku Cloud on 6h provides clear support/resistance zones and momentum signals.
Using 1d EMA200 as trend filter ensures alignment with daily trend, avoiding counter-trend trades.
Volume confirmation (1.5x 20-period average) filters weak breakouts.
Ichimoku performs well in both trending and ranging markets due to its multi-component design.
Targets 50-150 trades over 4 years (12-37/year) to balance opportunity with fee minimization.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8475_6h_ichimoku_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_PERIOD = 9      # Conversion line
KIJUN_PERIOD = 26      # Base line
SENKOU_B_PERIOD = 52   # Leading Span B
CHIKOU_SHIFT = 26      # Lagging span
TREND_PERIOD = 200     # 1d EMA for trend filter
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku Cloud components"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() + 
                  pd.Series(low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() + 
                 pd.Series(low).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(CHIKOU_SHIFT)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b = ((pd.Series(high).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).max() + 
                 pd.Series(low).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).min()) / 2).shift(CHIKOU_SHIFT)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods back
    chikou_span = pd.Series(close).shift(-CHIKOU_SHIFT)
    
    return tenkan_sen.values, kijun_sen.values, senkou_a.values, senkou_b.values, chikou_span.values

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=TREND_PERIOD, adjust=False, min_periods=TREND_PERIOD).mean().values
    
    # Price relative to 1d EMA200: above = bullish bias, below = bearish bias
    price_vs_ema = np.where(close_1d > ema_1d, 1, 
                     np.where(close_1d < ema_1d, -1, 0))  # 1=bullish, -1=bearish, 0=at EMA
    price_vs_ema_aligned = align_htf_to_ltf(prices, df_1d, price_vs_ema)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Ichimoku Cloud
    tenkan, kijun, senkou_a, senkou_b, chikou = calculate_ichimoku(high, low, close)
    
    # Cloud top and bottom (Senkou Span A and B)
    cloud_top = np.maximum(senkou_a, senkou_b)
    cloud_bottom = np.minimum(senkou_a, senkou_b)
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of all lookbacks)
    start = max(TREND_PERIOD, SENKOU_B_PERIOD + CHIKOU_SHIFT, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_vs_ema_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine market bias from 1d EMA200
        bull_bias = price_vs_ema_aligned[i] == 1   # 1d price above EMA200
        bear_bias = price_vs_ema_aligned[i] == -1  # 1d price below EMA200
        
        # Ichimoku signals
        # Price above cloud = bullish, below cloud = bearish
        price_above_cloud = close[i] > cloud_top[i] if not np.isnan(cloud_top[i]) else False
        price_below_cloud = close[i] < cloud_bottom[i] if not np.isnan(cloud_bottom[i]) else False
        
        # TK Cross: Tenkan crosses above/below Kijun
        tk_cross_up = (tenkan[i-1] <= kijun[i-1]) and (tenkan[i] > kijun[i]) if i > 0 and not np.isnan(tenkan[i-1]) and not np.isnan(kijun[i-1]) else False
        tk_cross_down = (tenkan[i-1] >= kijun[i-1]) and (tenkan[i] < kijun[i]) if i > 0 and not np.isnan(tenkan[i-1]) and not np.isnan(kijun[i-1]) else False
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions
        # Long: bullish bias + price above cloud + TK cross up + volume
        long_entry = bull_bias and price_above_cloud and tk_cross_up and volume_confirmed
        # Short: bearish bias + price below cloud + TK cross down + volume
        short_entry = bear_bias and price_below_cloud and tk_cross_down and volume_confirmed
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals