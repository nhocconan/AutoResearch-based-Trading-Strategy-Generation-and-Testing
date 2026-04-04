#!/usr/bin/env python3
"""
exp_6699_6h_ichimoku_cloud_12h_v1
Hypothesis: 6h Ichimoku cloud breakout with 12h trend filter and volume confirmation.
Ichimoku provides dynamic support/resistance (cloud), momentum (TK cross), and trend strength.
Using 12h timeframe for trend filter avoids counter-trend trades in bear markets like 2022.
Volume confirmation reduces false breakouts. Designed for 6h to capture swings with ~15-35 trades/year.
Works in bull (breakouts with volume) and bear (mean reversion at cloud edges during ranging).
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6699_6h_ichimoku_cloud_12h_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TK_PERIOD = 9      # Tenkan-sen (Conversion Line)
KJ_PERIOD = 26     # Kijun-sen (Base Line)
SSA_PERIOD = 52    # Senkou Span A (Leading Span A)
SSB_PERIOD = 26    # Senkou Span B (Leading Span B)
DISPLACEMENT = 26  # Kumo (cloud) displacement forward
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 6  # ~1.5 days (6h bars)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 12h for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend filter (above/below EMA50)
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate LTF Ichimoku components
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Tenkan-sen (Conversion Line): (HH + LL)/2 for TK_PERIOD
    def rolling_max(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).max().values
    def rolling_min(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).min().values
    
    tk_period_high = rolling_max(high, TK_PERIOD)
    tk_period_low = rolling_min(low, TK_PERIOD)
    tenkan_sen = (tk_period_high + tk_period_low) / 2.0
    
    # Kijun-sen (Base Line): (HH + LL)/2 for KJ_PERIOD
    kj_period_high = rolling_max(high, KJ_PERIOD)
    kj_period_low = rolling_min(low, KJ_PERIOD)
    kijun_sen = (kj_period_high + kj_period_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 displaced forward
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2.0)
    
    # Senkou Span B (Leading Span B): (HH + LL)/2 for SSB_PERIOD displaced forward
    ssb_period_high = rolling_max(high, SSB_PERIOD)
    ssb_period_low = rolling_min(low, SSB_PERIOD)
    senkou_span_b = (ssb_period_high + ssb_period_low) / 2.0
    
    # Align Ichimoku components (already calculated on LTF, no HTF alignment needed)
    # But we need to shift the leading spans forward by DISPLACEMENT periods
    # The cloud is plotted DISPLACEMENT periods ahead, so we use values from DISPLACEMENT bars ago
    senkou_span_a_leading = np.roll(senkou_span_a, -DISPLACEMENT)
    senkou_span_b_leading = np.roll(senkou_span_b, -DISPLACEMENT)
    # Fill the displaced values at the end with NaN (no future data)
    senkou_span_a_leading[-DISPLACEMENT:] = np.nan
    senkou_span_b_leading[-DISPLACEMENT:] = np.nan
    
    # Current cloud (Senkou Span A/B from DISPLACEMENT periods ago)
    senkou_span_a_current = np.roll(senkou_span_a, DISPLACEMENT)
    senkou_span_b_current = np.roll(senkou_span_b, DISPLACEMENT)
    senkou_span_a_current[:DISPLACEMENT] = np.nan
    senkou_span_b_current[:DISPLACEMENT] = np.nan
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period (need enough for Ichimoku calculation)
    start = max(TK_PERIOD, KJ_PERIOD, SSA_PERIOD, SSB_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + DISPLACEMENT
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if indicators not ready
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_span_a_leading[i]) or np.isnan(senkou_span_b_leading[i]) or
            np.isnan(ema_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
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
            
        # Determine trend from 12h EMA50
        uptrend_12h = close[i] > ema_aligned[i]
        downtrend_12h = close[i] < ema_aligned[i]
        
        # Cloud boundaries (leading spans)
        upper_cloud = np.maximum(senkou_span_a_leading[i], senkou_span_b_leading[i])
        lower_cloud = np.minimum(senkou_span_a_leading[i], senkou_span_b_leading[i])
        
        # TK cross
        tk_cross_up = tenkan_sen[i] > kijun_sen[i] and tenkan_sen[i-1] <= kijun_sen[i-1]
        tk_cross_down = tenkan_sen[i] < kijun_sen[i] and tenkan_sen[i-1] >= kijun_sen[i-1]
        
        # Price relative to cloud
        price_above_cloud = close[i] > upper_cloud
        price_below_cloud = close[i] < lower_cloud
        price_in_cloud = (close[i] >= lower_cloud) and (close[i] <= upper_cloud)
        
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Long conditions: TK cross up + price above/below cloud + volume + 12h uptrend
        long_signal = tk_cross_up and vol_confirmed and uptrend_12h and (
            price_above_cloud or  # Breakout above cloud
            (price_in_cloud and tenkan_sen[i] > kijun_sen[i])  # Pullback to TK in uptrend
        )
        
        # Short conditions: TK cross down + price above/below cloud + volume + 12h downtrend
        short_signal = tk_cross_down and vol_confirmed and downtrend_12h and (
            price_below_cloud or  # Breakdown below cloud
            (price_in_cloud and tenkan_sen[i] < kijun_sen[i])  # Pullback to TK in downtrend
        )
        
        # Enter new positions only if flat
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_signal:
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