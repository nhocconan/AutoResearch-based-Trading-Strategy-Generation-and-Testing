#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot long/short with 1d trend filter and volume confirmation
# - Camarilla pivots calculated from 1d OHLC: H4, L4, H3, L3 levels
# - Long when price crosses above H3 with 1d uptrend (close > EMA50) and volume spike
# - Short when price crosses below L3 with 1d downtrend (close < EMA50) and volume spike
# - Uses 12h timeframe targeting 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
# - 1d EMA50 filter ensures trading with higher timeframe trend direction
# - 12h volume > 1.5x 20-period average confirms breakout strength
# - Discrete position sizing (0.25) to minimize fee churn
# - ATR-based stoploss: exit when price moves against position by 2.0x ATR(14) or Camarilla level violated

name = "12h_1d_camarilla_volume_trend_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d ATR(14) for stoploss
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14_1d = np.zeros_like(tr)
    atr_14_1d[14-1] = np.mean(tr[:14])
    for i in range(14, len(tr)):
        atr_14_1d[i] = (atr_14_1d[i-1] * (14-1) + tr[i]) / 14
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # 1d Camarilla pivot levels (based on previous day's OHLC)
    # Camarilla: H4 = close + 1.5*(high-low), L4 = close - 1.5*(high-low)
    #           H3 = close + 1.125*(high-low), L3 = close - 1.125*(high-low)
    #           H2 = close + 0.75*(high-low), L2 = close - 0.75*(high-low)
    #           H1 = close + 0.5*(high-low), L1 = close - 0.5*(high-low)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # First bar uses current values
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    camarilla_h3 = prev_close + 1.125 * (prev_high - prev_low)
    camarilla_l3 = prev_close - 1.125 * (prev_high - prev_low)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # 1d volume confirmation: > 1.5x 20-period average
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * avg_volume_20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # 12h indicators
    close_12h = prices['close'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    
    # 12h ATR(14) for tighter stoploss
    tr1_12h = high_12h - low_12h
    tr2_12h = np.abs(high_12h - np.roll(close_12h, 1))
    tr3_12h = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    tr_12h[0] = tr1_12h[0]
    atr_14_12h = np.zeros_like(tr_12h)
    atr_14_12h[14-1] = np.mean(tr_12h[:14])
    for i in range(14, len(tr_12h)):
        atr_14_12h[i] = (atr_14_12h[i-1] * (14-1) + tr_12h[i]) / 14
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(atr_14_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ATR-based stoploss or price below L3 (Camarilla support)
            if (close_12h[i] < entry_price - 2.0 * entry_atr or 
                close_12h[i] < camarilla_l3_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ATR-based stoploss or price above H3 (Camarilla resistance)
            if (close_12h[i] > entry_price + 2.0 * entry_atr or 
                close_12h[i] > camarilla_h3_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakout with trend and volume filters
            if vol_spike_1d_aligned[i]:
                # Long signal: price breaks above H3 with 1d uptrend
                if close_12h[i] > camarilla_h3_aligned[i] and close_12h[i] > ema_50_1d_aligned[i]:
                    position = 1
                    entry_price = close_12h[i]
                    entry_atr = atr_14_12h[i]
                    signals[i] = 0.25
                # Short signal: price breaks below L3 with 1d downtrend
                elif close_12h[i] < camarilla_l3_aligned[i] and close_12h[i] < ema_50_1d_aligned[i]:
                    position = -1
                    entry_price = close_12h[i]
                    entry_atr = atr_14_12h[i]
                    signals[i] = -0.25
    
    return signals