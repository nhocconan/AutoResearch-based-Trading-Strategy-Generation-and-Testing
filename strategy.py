#!/usr/bin/env python3
"""
12h_HTF_Trend_LTF_Pullback_v1
Hypothesis: On 12h timeframe, use 1-week EMA(34) for trend direction and 1-day RSI(14) for pullback entries in the direction of the weekly trend. Volume confirmation (1.5x 20-period average) filters low-quality signals. Discrete sizing (0.25) and ATR-based stoploss (2.5x) control risk. Designed to work in both bull and bear markets by only trading with the weekly trend, reducing whipsaw. Target: 50-150 total trades over 4 years for BTC/ETH/SOL.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')   # Weekly trend
    df_1d = get_htf_data(prices, '1d')   # Daily for RSI and volume context
    
    if len(df_1w) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1-week EMA34 for trend regime ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === 1-day RSI14 for pullback signals ===
    close_1d = df_1d['close'].values
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_14_1d = 100 - (100 / (1 + rs))
    rsi_14_1d = rsi_14_1d.fillna(50).values  # Neutral when undefined
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # === 1-day volume confirmation (volume > 1.5x 20-period average) ===
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirmed_1d = vol_1d > (1.5 * vol_ma_20_1d)
    volume_confirmed_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_confirmed_1d.astype(float))
    
    # === 12h ATR (15-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=15, min_periods=15).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(rsi_14_1d_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_confirmed_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        ema_34_1w_val = ema_34_1w_aligned[i]
        rsi_14_1d_val = rsi_14_1d_aligned[i]
        vol_conf = volume_confirmed_1d_aligned[i] > 0.5  # Convert back to boolean
        
        # Trend regime from weekly EMA
        is_bull_trend = price > ema_34_1w_val
        is_bear_trend = price < ema_34_1w_val
        
        if position == 0:
            if is_bull_trend:
                # In bull trend: look for RSI pullback to oversold (<40) for long
                long_condition = (rsi_14_1d_val < 40) and vol_conf
                short_condition = False  # No shorts in bull trend
            else:  # bear trend
                # In bear trend: look for RSI pullback to overbought (>60) for short
                short_condition = (rsi_14_1d_val > 60) and vol_conf
                long_condition = False   # No longs in bear trend
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
                bars_since_entry = 0
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
                bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Minimum holding period of 8 bars to reduce churn
            if bars_since_entry < 8:
                signals[i] = 0.25 if position == 1 else -0.25
                continue
            
            # Check stoploss (2.5x ATR)
            if position == 1:
                if price < entry_price - 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit if RSI shows overextension (>70) in bull trend
                elif rsi_14_1d_val > 70:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > entry_price + 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit if RSI shows overextension (<30) in bear trend
                elif rsi_14_1d_val < 30:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_HTF_Trend_LTF_Pullback_v1"
timeframe = "12h"
leverage = 1.0