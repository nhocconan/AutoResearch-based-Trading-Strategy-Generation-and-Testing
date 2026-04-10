#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h volume confirmation and 1d trend filter
# - Long when price breaks above Camarilla H3 level with 4h volume spike and 1d uptrend (close > EMA50)
# - Short when price breaks below Camarilla L3 level with 4h volume spike and 1d downtrend (close < EMA50)
# - Uses 1h timeframe targeting 15-37 trades/year (60-150 total over 4 years) to minimize fee drag
# - 4h volume > 1.5x 20-period average confirms breakout strength
# - 1d EMA50 filter ensures trading with daily trend direction
# - Discrete position sizing (0.20) to minimize fee churn
# - ATR-based stoploss: exit when price moves against position by 2.0x ATR(14)

name = "1h_4h_1d_camarilla_volume_trend_atr_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 20 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 4h volume confirmation
    volume_4h = df_4h['volume'].values
    avg_volume_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_spike_4h = volume_4h > (1.5 * avg_volume_20_4h)
    vol_spike_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_spike_4h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_spike_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla pivots for previous period (using 1h data)
        # Pivots based on previous bar's high, low, close
        if i == 0:
            signals[i] = 0.0
            continue
            
        high_prev = prices['high'].iloc[i-1]
        low_prev = prices['low'].iloc[i-1]
        close_prev = prices['close'].iloc[i-1]
        
        # Camarilla pivot levels
        pivot = (high_prev + low_prev + close_prev) / 3
        range_prev = high_prev - low_prev
        
        # H3 and L3 levels (most significant for breakouts)
        camarilla_h3 = close_prev + range_prev * 1.1 / 4
        camarilla_l3 = close_prev - range_prev * 1.1 / 4
        
        if position == 1:  # Long position
            # Exit: ATR-based stoploss or price breaks below L3 (trend reversal)
            atr_14 = calculate_atr(prices['high'].values, prices['low'].values, prices['close'].values, 14)
            if (prices['close'].iloc[i] < entry_price - 2.0 * atr_14[i] or 
                prices['close'].iloc[i] < camarilla_l3):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: ATR-based stoploss or price breaks above H3 (trend reversal)
            atr_14 = calculate_atr(prices['high'].values, prices['low'].values, prices['close'].values, 14)
            if (prices['close'].iloc[i] > entry_price + 2.0 * atr_14[i] or 
                prices['close'].iloc[i] > camarilla_h3):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Look for Camarilla breakout with trend and volume filters
            if vol_spike_4h_aligned[i]:
                # Long signal: price breaks above H3 in daily uptrend
                if (prices['close'].iloc[i] > camarilla_h3 and 
                    prices['close'].iloc[i] > ema_50_1d_aligned[i]):
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    signals[i] = 0.20
                # Short signal: price breaks below L3 in daily downtrend
                elif (prices['close'].iloc[i] < camarilla_l3 and 
                      prices['close'].iloc[i] < ema_50_1d_aligned[i]):
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    signals[i] = -0.20
    
    return signals

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing (equivalent to RMA)"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = np.zeros_like(tr)
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr