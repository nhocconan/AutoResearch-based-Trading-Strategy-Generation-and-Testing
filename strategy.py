#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation
# - Long when price breaks above Donchian(20) high in 1w uptrend (close > EMA50) with volume spike
# - Short when price breaks below Donchian(20) low in 1w downtrend (close < EMA50) with volume spike
# - Uses discrete position sizing (0.25) to minimize fee churn
# - ATR-based stoploss: exit when price moves against position by 2.0x ATR(14)
# - Targets 20-50 trades/year (80-200 total over 4 years) to avoid fee drag
# - Works in bull via breakout continuation, in bear via short breakdowns

name = "1d_1w_donchian_breakout_volume_trend_atr_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w indicators
    close_1w = df_1w['close'].values
    
    # 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(100, n):
        # Need at least 20 bars for Donchian calculation
        if i < 20:
            continue
            
        # Calculate Donchian channels on 1d data
        lookback_start = i - 19  # 20 bars including current
        highest_high = prices['high'].iloc[lookback_start:i+1].max()
        lowest_low = prices['low'].iloc[lookback_start:i+1].min()
        
        # Skip if HTF trend data is invalid
        if np.isnan(ema_50_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ATR-based stoploss
            atr_14 = calculate_atr(prices, i, 14)
            if atr_14 is not None and prices['close'].iloc[i] < entry_price - 2.0 * atr_14:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ATR-based stoploss
            atr_14 = calculate_atr(prices, i, 14)
            if atr_14 is not None and prices['close'].iloc[i] > entry_price + 2.0 * atr_14:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check volume confirmation: volume > 1.5x 20-period average
            if i >= 20:
                avg_volume = prices['volume'].iloc[i-19:i+1].mean()
                volume_spike = prices['volume'].iloc[i] > (1.5 * avg_volume)
            else:
                volume_spike = False
            
            if volume_spike:
                # Long signal: price breaks above Donchian high in 1w uptrend
                if (prices['close'].iloc[i] > highest_high and 
                    prices['close'].iloc[i] > ema_50_1w_aligned[i]):
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    signals[i] = 0.25
                # Short signal: price breaks below Donchian low in 1w downtrend
                elif (prices['close'].iloc[i] < lowest_low and 
                      prices['close'].iloc[i] < ema_50_1w_aligned[i]):
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    signals[i] = -0.25
    
    return signals

def calculate_atr(prices, current_idx, period):
    """Calculate ATR using only data available up to current_idx"""
    if current_idx < period:
        return None
    
    # Calculate true range for each bar
    true_ranges = []
    for j in range(current_idx - period + 1, current_idx + 1):
        high = prices['high'].iloc[j]
        low = prices['low'].iloc[j]
        close_prev = prices['close'].iloc[j-1] if j > 0 else prices['close'].iloc[j]
        
        tr1 = high - low
        tr2 = abs(high - close_prev)
        tr3 = abs(low - close_prev)
        true_range = max(tr1, tr2, tr3)
        true_ranges.append(true_range)
    
    # Calculate ATR as average of true ranges
    return sum(true_ranges) / len(true_ranges) if true_ranges else None