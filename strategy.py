#!/usr/bin/env python3
"""
exp_6611_6h_ichimoku_cloud_1d_trend_v1
Hypothesis: 6h Ichimoku cloud breakout with 1d trend filter (EMA50/EMA200). Uses Ichimoku TK cross 
as entry signal with cloud as dynamic support/resistance. 1d EMA alignment ensures trading in 
direction of higher timeframe trend. Works in bull/bear markets by only taking longs when 1d 
trend is bullish (price > EMA50 > EMA200) and shorts when bearish (price < EMA50 < EMA200).
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6611_6h_ichimoku_cloud_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CONVERSION_PERIOD = 9   # Tenkan-sen
BASE_PERIOD = 26        # Kijun-sen
LEADING_SPAN_B_PERIOD = 52  # Senkou Span B
DISPLACEMENT = 26       # Kumo displacement
EMA_FAST = 50
EMA_SLOW = 200
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
MAX_HOLD_BARS = 12  # ~12 * 6h = 3 days

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA trend filter
    close_1d = df_1d['close'].values
    ema_fast_1d = pd.Series(close_1d).ewm(span=EMA_FAST, adjust=False).mean().values
    ema_slow_1d = pd.Series(close_1d).ewm(span=EMA_SLOW, adjust=False).mean().values
    
    # Align to LTF (6h) with shift(1) for completed bars only
    ema_fast_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_fast_1d)
    ema_slow_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_slow_1d)
    
    # Calculate LTF Ichimoku components
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Tenkan-sen (Conversion Line): (HH + LL)/2 for past 9 periods
    tenkan_sen = (pd.Series(high).rolling(window=CONVERSION_PERIOD, min_periods=CONVERSION_PERIOD).max() + 
                  pd.Series(low).rolling(window=CONVERSION_PERIOD, min_periods=CONVERSION_PERIOD).min()) / 2
    tenkan_sen = tenkan_sen.values
    
    # Kijun-sen (Base Line): (HH + LL)/2 for past 26 periods
    kijun_sen = (pd.Series(high).rolling(window=BASE_PERIOD, min_periods=BASE_PERIOD).max() + 
                 pd.Series(low).rolling(window=BASE_PERIOD, min_periods=BASE_PERIOD).min()) / 2
    kijun_sen = kijun_sen.values
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 displaced 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (HH + LL)/2 for past 52 periods displaced 26 periods ahead
    senkou_span_b = (pd.Series(high).rolling(window=LEADING_SPAN_B_PERIOD, min_periods=LEADING_SPAN_B_PERIOD).max() + 
                     pd.Series(low).rolling(window=LEADING_SPAN_B_PERIOD, min_periods=LEADING_SPAN_B_PERIOD).min()) / 2
    senkou_span_b = senkou_span_b.values
    
    # Current Kumo (cloud) boundaries - use values displaced back by 26 to align with price
    # Senkou Span A displaced 26 periods back = Senkou Span A[26:] shifted to current
    # Senkou Span B displaced 26 periods back = Senkou Span B[26:] shifted to current
    # For current cloud, we look at Senkou values that were calculated 26 periods ago
    upper_cloud = np.maximum(senkou_span_a, senkou_span_b)  # Will align properly in loop
    lower_cloud = np.minimum(senkou_span_a, senkou_span_b)  # Will align properly in loop
    
    # Align Ichimoku components to LTF with proper displacement
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)  # Dummy df_1d for alignment, will be overwritten
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    
    # Proper Ichimoku alignment - calculate on LTF directly
    tenkan_sen_ltf = (pd.Series(high).rolling(window=CONVERSION_PERIOD, min_periods=CONVERSION_PERIOD).max() + 
                      pd.Series(low).rolling(window=CONVERSION_PERIOD, min_periods=CONVERSION_PERIOD).min()) / 2
    kijun_sen_ltf = (pd.Series(high).rolling(window=BASE_PERIOD, min_periods=BASE_PERIOD).max() + 
                     pd.Series(low).rolling(window=BASE_PERIOD, min_periods=BASE_PERIOD).min()) / 2
    tenkan_sen_ltf = tenkan_sen_ltf.values
    kijun_sen_ltf = kijun_sen_ltf.values
    
    # Senkou Span A and B for cloud (calculated on LTF)
    senkou_span_a_ltf = ((tenkan_sen_ltf + kijun_sen_ltf) / 2)
    senkou_span_b_ltf = (pd.Series(high).rolling(window=LEADING_SPAN_B_PERIOD, min_periods=LEADING_SPAN_B_PERIOD).max() + 
                         pd.Series(low).rolling(window=LEADING_SPAN_B_PERIOD, min_periods=LEADING_SPAN_B_PERIOD).min()) / 2
    senkou_span_b_ltf = senkou_span_b_ltf.values
    
    # Current cloud boundaries (Senkou lines displaced 26 periods back)
    # For bar i, cloud is Senkou[A/B] from i+26
    upper_cloud_ltf = np.maximum(senkou_span_a_ltf, senkou_span_b_ltf)
    lower_cloud_ltf = np.minimum(senkou_span_a_ltf, senkou_span_b_ltf)
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Warmup: need enough data for Ichimoku calculations
    start = max(BASE_PERIOD, LEADING_SPAN_B_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + DISPLACEMENT + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if data not available
        if (np.isnan(ema_fast_1d_aligned[i]) or np.isnan(ema_slow_1d_aligned[i]) or 
            np.isnan(tenkan_sen_ltf[i]) or np.isnan(kijun_sen_ltf[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
                
        # Time-based exit
        if position != 0 and bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
            
        # Determine 1d trend bias
        # Bullish: price > EMA50 > EMA200
        # Bearish: price < EMA50 < EMA200
        bullish_bias = (close[i] > ema_fast_1d_aligned[i] > ema_slow_1d_aligned[i])
        bearish_bias = (close[i] < ema_fast_1d_aligned[i] < ema_slow_1d_aligned[i])
        
        # Ichimoku signals
        # TK Cross: Tenkan-sen crossing Kijun-sen
        tk_cross_up = tenkan_sen_ltf[i] > kijun_sen_ltf[i] and tenkan_sen_ltf[i-1] <= kijun_sen_ltf[i-1]
        tk_cross_down = tenkan_sen_ltf[i] < kijun_sen_ltf[i] and tenkan_sen_ltf[i-1] >= kijun_sen_ltf[i-1]
        
        # Price relative to cloud
        price_above_cloud = close[i] > upper_cloud_ltf[i]
        price_below_cloud = close[i] < lower_cloud_ltf[i]
        
        # Enter new positions only if flat
        if position == 0:
            # Long: TK cross up + price above cloud + bullish 1d trend + volume
            if (tk_cross_up and price_above_cloud and bullish_bias and 
                volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD):
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            # Short: TK cross down + price below cloud + bearish 1d trend + volume
            elif (tk_cross_down and price_below_cloud and bearish_bias and 
                  volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD):
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals