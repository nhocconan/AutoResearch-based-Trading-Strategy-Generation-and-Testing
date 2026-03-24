#!/usr/bin/env python3
"""
Experiment #491: 6h Primary + 1w/1d HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: 6h timeframe is underexplored (ZERO prior experiments). This strategy uses:
1. 1w HMA(21) = long-term trend bias (very slow, reduces whipsaws in bear markets)
2. 1d HMA(21) = medium-term confirmation (aligns with weekly bias)
3. 6h RSI(14) pullback = entry timing on dips in uptrend / rallies in downtrend
4. 6h Volume filter = taker_buy_ratio > 0.45 confirms long conviction
5. ATR(14)*2.5 stoploss = wider stops appropriate for 6h timeframe
6. LOOSE entry conditions to guarantee >=30 trades/year (target 40-60)

Key differences from failed 6h experiments:
- NO complex regime detection (Chop/ADX failed in #482, #484, #486, #488)
- NO weekly pivot logic (all pivot strategies failed - see "already tried" list)
- Single HTF chain: 1w → 1d → 6h (not dual HTF agreement which reduces trades)
- RSI thresholds 40/60 (not 30/70) for more frequent entries
- Volume confirmation reduces false breakouts

Target: Sharpe>0.45, trades>=120 train (30/year), trades>=15 test
Timeframe: 6h (middle ground between 4h and 12h)
Size: 0.25 base, 0.30 strong
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_hma_rsi_volume_1w1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_vr(close, volume, taker_buy_volume, period=14):
    """Volume Ratio - taker buy volume / total volume, smoothed"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    vr_raw = np.zeros(n)
    for i in range(n):
        if volume[i] > 1e-10:
            vr_raw[i] = taker_buy_volume[i] / volume[i]
        else:
            vr_raw[i] = 0.5
    
    vr = pd.Series(vr_raw).ewm(span=period, min_periods=period, adjust=False).mean().values
    return vr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1w HMA for long-term trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate and align 1d HMA for medium-term confirmation
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    hma_6h = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    vr = calculate_vr(close, volume, taker_buy_volume, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_6h[i]) or np.isnan(rsi[i]) or np.isnan(vr[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_50[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === 1w HTF LONG-TERM BIAS ===
        htf_weekly_bull = close[i] > hma_1w_aligned[i]
        htf_weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === 1d HTF MEDIUM-TERM CONFIRMATION ===
        htf_daily_bull = close[i] > hma_1d_aligned[i]
        htf_daily_bear = close[i] < hma_1d_aligned[i]
        
        # === 6h HMA TREND ===
        hma_bull = close[i] > hma_6h[i]
        hma_bear = close[i] < hma_6h[i]
        
        # === SMA FILTERS ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        above_sma200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_sma200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # === RSI CONDITIONS (LOOSE: 40/60) ===
        rsi_oversold = rsi[i] < 40.0
        rsi_overbought = rsi[i] > 60.0
        rsi_recovery = rsi[i] > 45.0 and rsi[i-1] <= 45.0
        rsi_weakness = rsi[i] < 55.0 and rsi[i-1] >= 55.0
        
        # === VOLUME CONFIRMATION ===
        vol_bull = vr[i] > 0.48  # More buyer conviction
        vol_bear = vr[i] < 0.42  # More seller conviction
        
        # === ENTRY LOGIC (LOOSE - OR logic for trade generation) ===
        desired_signal = 0.0
        
        # TREND LONG: Weekly bull + (Daily bull OR 6h HMA bull) + RSI pullback
        if htf_weekly_bull:
            if htf_daily_bull and hma_bull and rsi_oversold and vol_bull:
                desired_signal = SIZE_STRONG
            elif htf_daily_bull and rsi_recovery and above_sma50:
                desired_signal = SIZE_BASE
            elif hma_bull and rsi[i] > 40.0 and rsi[i-1] <= 40.0 and above_sma50:
                desired_signal = SIZE_BASE
        
        # TREND SHORT: Weekly bear + (Daily bear OR 6h HMA bear) + RSI rally
        elif htf_weekly_bear:
            if htf_daily_bear and hma_bear and rsi_overbought and vol_bear:
                desired_signal = -SIZE_STRONG
            elif htf_daily_bear and rsi_weakness and below_sma50:
                desired_signal = -SIZE_BASE
            elif hma_bear and rsi[i] < 60.0 and rsi[i-1] >= 60.0 and below_sma50:
                desired_signal = -SIZE_BASE
        
        # MEAN REVERSION LONG: RSI extreme oversold + above SMA200
        if desired_signal == 0.0:
            if rsi[i] < 35.0 and above_sma200:
                desired_signal = SIZE_BASE * 0.8
        
        # MEAN REVERSION SHORT: RSI extreme overbought + below SMA200
        if desired_signal == 0.0:
            if rsi[i] > 65.0 and below_sma200:
                desired_signal = -SIZE_BASE * 0.8
        
        # === STOPLOSS CHECK (2.5x ATR from entry for 6h timeframe) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        elif desired_signal >= SIZE_BASE * 0.5:
            final_signal = SIZE_BASE * 0.8
        elif desired_signal <= -SIZE_BASE * 0.5:
            final_signal = -SIZE_BASE * 0.8
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                # Set stoploss (2.5x ATR for 6h timeframe)
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
        
        signals[i] = final_signal
    
    return signals