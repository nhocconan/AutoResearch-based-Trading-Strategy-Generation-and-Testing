#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_Filter_Volume_Chop_Regime_v1
Hypothesis: On daily timeframe, KAMA direction (trend) combined with RSI extremes (mean reversion within trend) 
and volume confirmation captures sustainable moves. Choppiness index regime filter avoids sideways markets. 
ATR-based stoploss manages risk. Designed for low trade frequency (<30/year) to minimize fee drag in bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for trend regime)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1w EMA34 for weekly trend regime ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === Daily indicators ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA (adaptive trend)
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # 10-period sum of abs changes
    # Handle first 9 values
    change = np.concatenate([np.full(9, np.nan), change])
    volatility = np.concatenate([np.full(9, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # Initialize KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        if not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = close[i]
    
    # RSI(14)
    delta = np.diff(close)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    loss_ma = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(loss_ma != 0, gain_ma / loss_ma, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation (volume > 1.5x 20-day average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma_20)
    
    # Choppiness Index (14-period) for regime filter
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.where((hh - ll) != 0, 
                    100 * np.log10(tr_sum / (hh - ll)) / np.log10(14), 
                    50)
    
    # ATR(14) for stoploss
    tr_atr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr_atr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(50, n):  # warmup for indicators
        # Skip if indicators not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or np.isnan(volume_confirmed[i]) or 
            np.isnan(chop[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        ema_34_1w_val = ema_34_1w_aligned[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        vol_conf = volume_confirmed[i]
        chop_val = chop[i]
        atr_val = atr[i]
        
        # Trend regime from weekly EMA34
        is_bull = price > ema_34_1w_val
        is_bear = price < ema_34_1w_val
        
        # Choppiness filter: only trade when CHOP < 61.8 (trending) or > 38.2 (not too choppy)
        # Actually, we want to avoid extreme chop: CHOP > 61.8 is too choppy
        not_too_choppy = chop_val <= 61.8
        
        if position == 0:
            if is_bull and not_too_choppy:
                # Bull regime: long on pullbacks (RSI < 40) with volume
                long_condition = (price > kama_val) and (rsi_val < 40) and vol_conf
                # Short only on extreme overbought
                short_condition = (rsi_val > 80) and vol_conf and (price < kama_val * 0.98)
            elif is_bear and not_too_choppy:
                # Bear regime: short on bounces (RSI > 60) with volume
                short_condition = (price < kama_val) and (rsi_val > 60) and vol_conf
                # Long only on extreme oversold
                long_condition = (rsi_val < 20) and vol_conf and (price > kama_val * 1.02)
            else:
                long_condition = False
                short_condition = False
            
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
            
            # Minimum holding period of 3 days to reduce churn
            if bars_since_entry < 3:
                signals[i] = 0.25 if position == 1 else -0.25
                continue
            
            # Check stoploss (2.5x ATR)
            if position == 1:
                if price < entry_price - 2.5 * atr_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit if RSI shows exhaustion
                elif rsi_val > 70:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > entry_price + 2.5 * atr_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit if RSI shows exhaustion
                elif rsi_val < 30:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Direction_RSI_Filter_Volume_Chop_Regime_v1"
timeframe = "1d"
leverage = 1.0