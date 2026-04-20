# NOTE: This script is provided for educational and illustrative purposes only. It is not intended as financial advice or as a recommendation to engage in any specific trading or investment strategy. Users should conduct their own research and consult with a qualified financial advisor before making any investment decisions.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_Price_Action_Confluence_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h and 1d data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 2 or len(df_1d) < 2:
        return np.zeros(n)
    
    # === 12h: 20-period EMA for trend direction ===
    close_12h = df_12h['close'].values
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # === 1d: Calculate pivot points (standard) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # R1 = 2*P - L, S1 = 2*P - H
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    # R2 = P + (H - L), S2 = P - (H - L)
    r2_1d = pivot_1d + (high_1d - low_1d)
    s2_1d = pivot_1d - (high_1d - low_1d)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    r3_1d = high_1d + 2 * (pivot_1d - low_1d)
    s3_1d = low_1d - 2 * (high_1d - pivot_1d)
    
    # Align all pivot levels
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # === 6h: ATR(14) for volatility and stop loss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Get aligned values
        ema_trend = ema_20_12h_aligned[i]
        pivot = pivot_1d_aligned[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        r2 = r2_1d_aligned[i]
        s2 = s2_1d_aligned[i]
        r3 = r3_1d_aligned[i]
        s3 = s3_1d_aligned[i]
        current_atr = atr[i]
        current_close = prices['close'].iloc[i]
        current_volume = prices['volume'].iloc[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_trend) or np.isnan(pivot) or np.isnan(r1) or np.isnan(s1) or
            np.isnan(r2) or np.isnan(s2) or np.isnan(r3) or np.isnan(s3) or np.isnan(current_atr)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # === Volume condition: current volume > 1.3x 24-period 6h average volume ===
        if i >= 24:
            vol_ma = np.mean(prices['volume'].iloc[i-24:i].values)
            vol_condition = current_volume > 1.3 * vol_ma
        else:
            vol_condition = False
        
        if position == 0:
            # Long conditions:
            # 1. Price above 12h EMA20 (uptrend)
            # 2. Price breaks above R1 with volume
            # 3. Price is below R2 (not overextended)
            if (current_close > ema_trend and
                current_close > r1 and
                vol_condition and
                current_close < r2):
                signals[i] = 0.25
                position = 1
                entry_price = current_close
            
            # Short conditions:
            # 1. Price below 12h EMA20 (downtrend)
            # 2. Price breaks below S1 with volume
            # 3. Price is above S2 (not overextended)
            elif (current_close < ema_trend and
                  current_close < s1 and
                  vol_condition and
                  current_close > s2):
                signals[i] = -0.25
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Long exit conditions:
            # 1. Price falls below 12h EMA20 (trend change)
            # 2. Price hits S1 (strong support - take profit)
            # 3. ATR-based stop loss
            if (current_close < ema_trend or
                current_close <= s1 or
                current_close < entry_price - 2.5 * current_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit conditions:
            # 1. Price rises above 12h EMA20 (trend change)
            # 2. Price hits R1 (strong resistance - take profit)
            # 3. ATR-based stop loss
            if (current_close > ema_trend or
                current_close >= r1 or
                current_close > entry_price + 2.5 * current_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals