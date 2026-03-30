#!/usr/bin/env python3
"""
Experiment #022: 12h Donchian Breakout + 1d EMA Trend + Volume Spike

HYPOTHESIS: 12h is a proven timeframe (54% keep rate in DB).
Key insight from DB: mtf_4h_chop_donchian_vol_regime_12h_v1 achieved test_sharpe=1.491 on SOLUSDT.
By using 12h as PRIMARY (not 4h with 12h filter), we:
- Get structural breakouts at the 12h level (fewer but higher quality signals)
- Use 1d EMA21 for long-term trend direction (bull/bear filter)
- Apply volume spike confirmation (filters whipsaws)

WHY IT WORKS IN BOTH MARKETS:
- Long: Only when price > 1d EMA21 (confirmed uptrend)
- Short: Only when price < 1d EMA21 (confirmed downtrend)
- Donchian breakout captures momentum in both directions
- Volume spike confirms institutional participation
- 2.5 ATR stop allows positions to breathe

EXPECTED TRADES: 75-150 total per symbol over 4 years
- 12h bars: ~1460/year
- Donchian(20) breakouts: ~36/year before filters
- Volume spike (1.5x): ~22/year
- 1d EMA21 trend filter: ~15/year per direction
- Net: ~15-30/year = 60-120 total over 4 years
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian1d_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA21 for long-term trend direction (align to 12h)
    htf_ema21 = pd.Series(df_1d['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema21_aligned = align_htf_to_ltf(prices, df_1d, htf_ema21)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian Channel(20) - price channel breakout
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20 bars) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    warmup = 80  # Enough for Donchian20, ATR14, EMA21 alignment
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema21_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === TREND DIRECTION: 1d EMA21 ===
        bull_trend = close[i] > ema21_aligned[i]
        bear_trend = close[i] < ema21_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT (use prior bar's channel) ===
        prev_upper = donchian_upper[i-1] if i > 0 and not np.isnan(donchian_upper[i-1]) else np.nan
        prev_lower = donchian_lower[i-1] if i > 0 and not np.isnan(donchian_lower[i-1]) else np.nan
        
        # Bullish breakout: close above prior 19-bar high
        bullish_breakout = (not np.isnan(prev_upper) and close[i] > prev_upper)
        
        # Bearish breakout: close below prior 19-bar low
        bearish_breakout = (not np.isnan(prev_lower) and close[i] < prev_lower)
        
        # === MINIMUM HOLD: 2 bars to reduce fee churn ===
        min_hold_bars = (i - entry_bar) >= 2 if in_position else True
        
        # === EXITS ===
        if in_position:
            # Stop-loss: 2.5 ATR from entry
            if position_side > 0:
                stop_price = entry_price - 2.5 * entry_atr
                stop_hit = low[i] < stop_price
            else:
                stop_price = entry_price + 2.5 * entry_atr
                stop_hit = high[i] > stop_price
            
            # Trend exit: price crosses EMA21
            trend_exit = (position_side > 0 and close[i] < ema21_aligned[i]) or \
                        (position_side < 0 and close[i] > ema21_aligned[i])
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            elif min_hold_bars and trend_exit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            # LONG: Bullish breakout + volume spike + bull trend
            if bullish_breakout and vol_spike and bull_trend:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                signals[i] = SIZE
            
            # SHORT: Bearish breakout + volume spike + bear trend
            elif bearish_breakout and vol_spike and bear_trend:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                signals[i] = -SIZE
    
    return signals