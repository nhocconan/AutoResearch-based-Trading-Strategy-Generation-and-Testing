#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot long/short with 1d trend filter and volume confirmation
# - Long: price touches Camarilla L3 support + 1d EMA(50) uptrend + volume > 1.3x 20-period average
# - Short: price touches Camarilla H3 resistance + 1d EMA(50) downtrend + volume > 1.3x 20-period average
# - Uses discrete position sizing (0.25) to minimize fee churn
# - ATR-based stoploss (1.5x ATR(14)) to manage risk
# - Designed for 12h timeframe: targets 12-37 trades/year to avoid fee drag
# - Camarilla pivots work well in ranging markets (common in bear/consolidation)
# - Volume confirmation reduces false breakouts
# - 1d EMA filter ensures alignment with higher timeframe trend

name = "12h_1d_camarilla_pivot_trend_volume_v3"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Pre-compute 12h typical price for Camarilla calculation
    typical_price = (prices['high'] + prices['low'] + prices['close']) / 3
    tp_high = typical_price.rolling(window=20, min_periods=20).max().values
    tp_low = typical_price.rolling(window=20, min_periods=20).min().values
    tp_close = typical_price.values
    
    # Calculate Camarilla levels (based on previous day's range)
    # H4 = Close + 1.5*(High-Low), H3 = Close + 1.1*(High-Low), L3 = Close - 1.1*(High-Low), L4 = Close - 1.5*(High-Low)
    # For intraday, we use previous bar's high/low
    prev_high = prices['high'].shift(1).values
    prev_low = prices['low'].shift(1).values
    prev_close = prices['close'].shift(1).values
    
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low)
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low)
    
    # Pre-compute 12h volume confirmation
    volume_12h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_12h > (1.3 * avg_volume_20)
    
    # Pre-compute 12h ATR(14) for stoploss
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_spike[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below Camarilla L4 OR stoploss hit
            camarilla_l4 = prev_close[i] - 1.5 * (prev_high[i] - prev_low[i])
            if low_12h[i] < camarilla_l4 or close_12h[i] < entry_price - 1.5 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Camarilla H4 OR stoploss hit
            camarilla_h4 = prev_close[i] + 1.5 * (prev_high[i] - prev_low[i])
            if high_12h[i] > camarilla_h4 or close_12h[i] > entry_price + 1.5 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla touch with trend and volume filters
            if vol_spike[i]:
                # Long: price touches or goes below L3 + 1d EMA uptrend (close > EMA)
                if low_12h[i] <= camarilla_l3[i] and close_12h[i] > ema_50_aligned[i]:
                    position = 1
                    entry_price = close_12h[i]
                    signals[i] = 0.25
                # Short: price touches or goes above H3 + 1d EMA downtrend (close < EMA)
                elif high_12h[i] >= camarilla_h3[i] and close_12h[i] < ema_50_aligned[i]:
                    position = -1
                    entry_price = close_12h[i]
                    signals[i] = -0.25
    
    return signals