#!/usr/bin/env python3
"""
Experiment #008: 30m primary timeframe with 4h trend filter
Hypothesis: 4h HMA trend + 30m RSI pullback + ATR stoploss + BBW regime filter
- 4h HMA(21) defines macro trend (call ONCE before loop)
- 30m RSI(14) for entry timing on pullbacks
- Bollinger Band Width filters low-volatility chop
- ATR(14) trailing stoploss at 2.5*ATR
- Discrete signals: 0.0, ±0.25, ±0.35 to minimize fee churn
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_hma_rsi_bbw_atr_30m_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average for trend detection"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    diff = 2 * wma1 - wma2
    hma = diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility and stoploss"""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    tr1 = high_s - low_s
    tr2 = abs(high_s - close_s.shift(1))
    tr3 = abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    return atr.values

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    return rsi

def calculate_bbw(high, low, close, period=20):
    """Bollinger Band Width for regime detection"""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    bbw = (2 * std) / sma
    return bbw.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # === LOAD HTF DATA ONCE BEFORE LOOP ===
    df_4h = get_htf_data(prices, '4h')
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # === CALCULATE 30m INDICATORS ===
    rsi = calculate_rsi(close, 14)
    atr = calculate_atr(high, low, close, 14)
    bbw = calculate_bbw(high, low, close, 20)
    close_s = pd.Series(close)
    sma200 = close_s.rolling(window=200, min_periods=200).mean().values
    
    # === BBW PERCENTILE FOR REGIME FILTER ===
    bbw_percentile = pd.Series(bbw).rolling(window=100, min_periods=50).apply(
        lambda x: np.percentile(x[~np.isnan(x)], 30) if len(x[~np.isnan(x)]) > 0 else np.nan
    ).values
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_MAX = 0.35
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(rsi[i]) or np.isnan(atr[i]) or np.isnan(bbw[i]):
            signals[i] = 0.0
            position_side = 0
            continue
        
        # === TREND FILTER (4h HMA) ===
        trend_4h = 1.0 if hma_4h_aligned[i] > 0 and close[i] > hma_4h_aligned[i] else -1.0
        
        # === REGIME FILTER (BBW) ===
        # Only trade when volatility is above 30th percentile (avoid dead chop)
        regime_ok = bbw[i] > bbw_percentile[i] if not np.isnan(bbw_percentile[i]) else True
        
        # === LONG SETUP ===
        if trend_4h > 0 and regime_ok:
            # RSI pullback entry (not overbought)
            if rsi[i] < 55 and rsi[i] > 35:
                # Confirm above 200 SMA for strong trend
                if close[i] > sma200[i]:
                    if position_side <= 0:
                        signals[i] = SIZE_ENTRY
                        position_side = 1
                        entry_price = close[i]
                        highest_since_entry = close[i]
                    else:
                        signals[i] = SIZE_MAX  # add to position
            elif rsi[i] >= 55:
                signals[i] = SIZE_ENTRY if position_side > 0 else 0.0  # hold or flat
        
        # === SHORT SETUP ===
        elif trend_4h < 0 and regime_ok:
            # RSI pullback entry (not oversold)
            if rsi[i] > 45 and rsi[i] < 65:
                # Confirm below 200 SMA for strong trend
                if close[i] < sma200[i]:
                    if position_side >= 0:
                        signals[i] = -SIZE_ENTRY
                        position_side = -1
                        entry_price = close[i]
                        lowest_since_entry = close[i]
                    else:
                        signals[i] = -SIZE_MAX  # add to position
            elif rsi[i] <= 45:
                signals[i] = -SIZE_ENTRY if position_side < 0 else 0.0  # hold or flat
        else:
            signals[i] = 0.0
            position_side = 0
        
        # === STOPLOSS LOGIC (2.5 * ATR) ===
        if position_side == 1:
            highest_since_entry = max(highest_since_entry, close[i])
            stoploss_price = entry_price - 2.5 * atr[i]
            trail_stop = highest_since_entry - 2.5 * atr[i]
            effective_stop = max(stoploss_price, trail_stop)
            
            if close[i] < effective_stop:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
        
        elif position_side == -1:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stoploss_price = entry_price + 2.5 * atr[i]
            trail_stop = lowest_since_entry + 2.5 * atr[i]
            effective_stop = min(stoploss_price, trail_stop)
            
            if close[i] > effective_stop:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
        
        # === TAKE PROFIT (reduce at 2R) ===
        if position_side == 1 and signals[i] != 0.0:
            profit_target = entry_price + 2 * 2.5 * atr[i]  # 2R = 2 * stoploss distance
            if close[i] > profit_target:
                signals[i] = SIZE_ENTRY  # reduce from SIZE_MAX to SIZE_ENTRY
        
        elif position_side == -1 and signals[i] != 0.0:
            profit_target = entry_price - 2 * 2.5 * atr[i]
            if close[i] < profit_target:
                signals[i] = -SIZE_ENTRY  # reduce from -SIZE_MAX to -SIZE_ENTRY
    
    return signals