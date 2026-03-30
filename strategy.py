#!/usr/bin/env python3
"""
Experiment #004: 4h Donchian Breakout + 1w EMA Trend + Volume Confirmation

HYPOTHESIS: 
- 4h timeframe: proven to generate 75-200 trades/symbol (1d fails with <50 trades)
- 1w EMA21: provides clean multi-week trend direction (bull/bear regime)
- 4h Donchian(20): captures structural breakouts with established ranges
- Volume spike 1.8x: filters false breakouts (key from DB winners)
- 4-bar minimum hold: reduces fee churn from whipsaws

WHY IT SHOULD WORK IN BOTH MARKETS:
- Donchian breakout works in bull breakouts AND bear breakdown crashes
- 1w trend filter prevents long entries in bear regime (avoids 2022 whipsaw)
- Volume confirmation filters low-volume false breakouts
- ATR-based stoploss adapts to volatility regime

EXPECTED TRADES: 75-150 per symbol over 4 years (18-37/year)
- Donchian(20) on 4h: ~1 breakout per 30-40 bars
- Volume 1.8x filter: reduces by ~35%
- 1w trend filter: reduces by ~25%
- Final: ~75-150/symbol (safe range, >50 minimum threshold)

KEY FIX FROM RECENT FAILURES:
- #014/012: missed volume confirmation → negative Sharpe
- #015: volume too loose (311 trades) → fee drag
- This version: 1.8x volume + 4-bar min hold + 1w trend = tighter entry
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_1w_ema_v1"
timeframe = "4h"
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
    
    # === Load 1w data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA21 for trend direction (align to 4h)
    htf_ema = pd.Series(df_1w['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, htf_ema)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian Channel(20) - established price range
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
    
    # Warmup: 100 bars for all indicators to stabilize
    warmup = 100
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === 1w TREND DIRECTION ===
        bull_trend = close[i] > ema_1w_aligned[i]
        bear_trend = close[i] < ema_1w_aligned[i]
        
        # === VOLUME CONFIRMATION (1.8x = tighter than loose 1.5x) ===
        vol_spike = vol_ratio[i] > 1.8
        
        # === DONCHIAN BREAKOUT using PRIOR bar's channel ===
        prev_high_19 = donchian_upper[i-1] if i > 0 and not np.isnan(donchian_upper[i-1]) else np.nan
        prev_low_19 = donchian_lower[i-1] if i > 0 and not np.isnan(donchian_lower[i-1]) else np.nan
        
        # Bullish breakout: close above prior 19-bar high
        bullish_breakout = (not np.isnan(prev_high_19) and close[i] > prev_high_19)
        
        # Bearish breakout: close below prior 19-bar low
        bearish_breakout = (not np.isnan(prev_low_19) and close[i] < prev_low_19)
        
        # Minimum hold: 4 bars (16h) to reduce fee churn
        min_hold_bars = (i - entry_bar) >= 4 if in_position else True
        
        # === EXITS ===
        if in_position:
            # Stop-loss: 2.5 ATR from entry
            if position_side > 0:
                stop_price = entry_price - 2.5 * entry_atr
                stop_hit = low[i] < stop_price
            else:
                stop_price = entry_price + 2.5 * entry_atr
                stop_hit = high[i] > stop_price
            
            # Trend exit: price crosses 1w EMA
            trend_exit = (position_side > 0 and close[i] < ema_1w_aligned[i]) or \
                        (position_side < 0 and close[i] > ema_1w_aligned[i])
            
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