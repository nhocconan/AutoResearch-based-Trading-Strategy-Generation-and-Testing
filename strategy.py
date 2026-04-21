#!/usr/bin/env python3
"""
1h_RSI_Divergence_Volume_Regime_Filter
Hypothesis: On 1h timeframe, use 4h trend direction (EMA50) and 1d regime filter (Choppiness Index) to filter RSI divergence signals.
Long when: price makes lower low, RSI makes higher low (bullish divergence), price > 4h EMA50, and chop < 61.8 (trending regime).
Short when: price makes higher high, RSI makes lower high (bearish divergence), price < 4h EMA50, and chop < 61.8 (trending regime).
Add volume confirmation: current volume > 1.5x 20-period MA.
Designed for low trade frequency (~20-40/year) with regime filter to avoid whipsaws in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 60 or len(df_1d) < 60:
        return np.zeros(n)
    
    # === 4h EMA50 for trend filter ===
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # === 1d Choppiness Index for regime filter (14-period) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(TR)/ (HH - LL)) / log10(14)
    sum_tr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    chop_1d = 100 * (np.log10(sum_tr_1d) - np.log10(np.maximum(hh_1d - ll_1d, 1e-10))) / np.log10(14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === 1h RSI (14-period) for divergence detection ===
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.maximum(avg_loss, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === Volume confirmation (1.5x 20-period MA) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Session filter: 08-20 UTC ===
    hours = prices.index.hour  # open_time is already datetime64[ms]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(chop_1d_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        ema_50_4h_val = ema_50_4h_aligned[i]
        chop_val = chop_1d_aligned[i]
        vol_avg = vol_ma[i]
        hour = hours[i]
        
        # Session filter: only trade 08-20 UTC
        in_session = (8 <= hour <= 20)
        
        # Volume confirmation
        volume_confirm = volume_now > 1.5 * vol_avg
        
        # RSI divergence detection (need 3 bars lookback)
        if i >= 2:
            # Bullish divergence: price lower low, RSI higher low
            bull_div = (close[i] < close[i-2]) and (rsi[i] > rsi[i-2])
            # Bearish divergence: price higher high, RSI lower high
            bear_div = (close[i] > close[i-2]) and (rsi[i] < rsi[i-2])
        else:
            bull_div = False
            bear_div = False
        
        if position == 0:
            if in_session and volume_confirm:
                # Long: bullish divergence, price above 4h EMA50, trending regime (chop < 61.8)
                long_condition = bull_div and (price > ema_50_4h_val) and (chop_val < 61.8)
                # Short: bearish divergence, price below 4h EMA50, trending regime (chop < 61.8)
                short_condition = bear_div and (price < ema_50_4h_val) and (chop_val < 61.8)
                
                if long_condition:
                    signals[i] = 0.20
                    position = 1
                    entry_price = price
                    bars_since_entry = 0
                elif short_condition:
                    signals[i] = -0.20
                    position = -1
                    entry_price = price
                    bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Minimum holding period of 6 bars to reduce churn
            if bars_since_entry < 6:
                signals[i] = 0.20 if position == 1 else -0.20
                continue
            
            # Exit conditions: RSI crosses opposite extreme or regime changes to ranging
            if position == 1:
                # Exit long on bearish divergence or chop > 61.8 (ranging) or RSI > 70
                if bear_div or (chop_val > 61.8) or (rsi[i] > 70):
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                # Exit short on bullish divergence or chop > 61.8 (ranging) or RSI < 30
                if bull_div or (chop_val > 61.8) or (rsi[i] < 30):
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_RSI_Divergence_Volume_Regime_Filter"
timeframe = "1h"
leverage = 1.0