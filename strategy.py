#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h mean reversion at weekly pivot levels with volume confirmation
# In ranging markets (common in BTC/ETH 2025), price tends to revert to weekly pivot points
# Volume spikes confirm institutional interest at these key levels
# Using 12h timeframe keeps trade frequency low (target: 20-40 trades/year) to minimize fee drag
# Works in both bull/bear markets as mean reversion occurs in all regimes

name = "12h_WeeklyPivot_MeanReversion_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # === Weekly Pivot Points (standard calculation) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot = (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # R1 = 2*P - L, S1 = 2*P - H
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    # R2 = P + (H - L), S2 = P - (H - L)
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    r3_1w = high_1w + 2 * (pivot_1w - low_1w)
    s3_1w = low_1w - 2 * (high_1w - pivot_1w)
    
    # Align weekly pivot levels to 12h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # === 12h: ATR(14) for volatility and stop loss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === 12h: 20-period EMA for trend filter (avoid trading against strong trends) ===
    close_12h = close  # Already 12h data
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Get aligned values
        pivot = pivot_1w_aligned[i]
        r1 = r1_1w_aligned[i]
        s1 = s1_1w_aligned[i]
        r2 = r2_1w_aligned[i]
        s2 = s2_1w_aligned[i]
        r3 = r3_1w_aligned[i]
        s3 = s3_1w_aligned[i]
        ema_trend = ema_20_12h[i]
        current_atr = atr[i]
        current_close = prices['close'].iloc[i]
        current_volume = prices['volume'].iloc[i]
        
        # Skip if any value is NaN
        if (np.isnan(pivot) or np.isnan(r1) or np.isnan(s1) or np.isnan(r2) or np.isnan(s2) or
            np.isnan(r3) or np.isnan(s3) or np.isnan(ema_trend) or np.isnan(current_atr)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # === Volume condition: current volume > 1.5x 20-period 12h average volume ===
        if i >= 20:
            vol_ma = np.mean(prices['volume'].iloc[i-20:i].values)
            vol_condition = current_volume > 1.5 * vol_ma
        else:
            vol_condition = False
        
        if position == 0:
            # Long conditions:
            # 1. Price touches or goes below S1 with volume (support test)
            # 2. Not in strong downtrend (price above EMA20 or at least not far below)
            # 3. Not at extreme oversold (above S2 to avoid catching falling knives)
            if (current_close <= s1 and
                vol_condition and
                current_close > ema_trend * 0.98 and  # Allow small tolerance below EMA
                current_close > s2):
                signals[i] = 0.25
                position = 1
                entry_price = current_close
            
            # Short conditions:
            # 1. Price touches or goes above R1 with volume (resistance test)
            # 2. Not in strong uptrend (price below EMA20 or at least not far above)
            # 3. Not at extreme overbought (below R2 to avoid buying tops)
            elif (current_close >= r1 and
                  vol_condition and
                  current_close < ema_trend * 1.02 and  # Allow small tolerance above EMA
                  current_close < r2):
                signals[i] = -0.25
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Long exit conditions:
            # 1. Price reaches pivot (mean reversion target)
            # 2. Price breaks below S2 (failed support, go short instead)
            # 3. ATR-based stop loss
            if (current_close >= pivot or
                current_close < s2 or
                current_close < entry_price - 2.5 * current_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit conditions:
            # 1. Price reaches pivot (mean reversion target)
            # 2. Price breaks above R2 (failed resistance, go long instead)
            # 3. ATR-based stop loss
            if (current_close <= pivot or
                current_close > r2 or
                current_close > entry_price + 2.5 * current_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals