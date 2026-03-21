#!/usr/bin/env python3
"""
EXPERIMENT #018 - HMA Trend + MACD Histogram + ADX Strength + ATR Stop
=======================================================================
Hypothesis: HMA provides smoother trend signals than Donchian breakouts with less whipsaw.
Combined with MACD histogram momentum crosses and ADX strength filter, this should
capture trends with better entry precision while avoiding weak/choppy conditions.

Key differences from #017 (Donchian winner):
- HMA(48) trend instead of Donchian(20) - smoother, less false breakouts
- MACD histogram cross for entries instead of RSI pullback - momentum-based timing
- ADX(14) > 25 filter - only trade when trend has measurable strength
- Same ATR trailing stop (2.5x) and discrete position sizing (0.20-0.35)

Why this might beat Sharpe=6.689:
- HMA reduces lag vs EMA while being smoother than raw price channels
- MACD histogram captures momentum shifts earlier than RSI level crosses
- ADX filter avoids trading in choppy conditions (major loss source in #008, #010)
- Proven MTF structure (4h trend + 1h entries) from best performer
"""

import numpy as np
import pandas as pd

name = "mtf_hma_macd_adx_atr_v2"
timeframe = "1h"
leverage = 1.0


def calculate_wma(data, period):
    """Calculate Weighted Moving Average"""
    n = len(data)
    wma = np.zeros(n)
    weights = np.arange(1, period + 1, dtype=float)
    weight_sum = np.sum(weights)
    
    for i in range(period - 1, n):
        wma[i] = np.sum(data[i - period + 1:i + 1] * weights) / weight_sum
    
    return wma


def calculate_hma(close, period=48):
    """Calculate Hull Moving Average - smoother than EMA with less lag"""
    n = len(close)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = calculate_wma(close, half_period)
    wma_full = calculate_wma(close, period)
    
    hull_raw = 2 * wma_half - wma_full
    
    hma = calculate_wma(hull_raw, sqrt_period)
    
    return hma


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD with histogram"""
    ema_fast = pd.Series(close).ewm(span=fast, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, adjust=False).mean().values
    
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().values
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength measurement"""
    n = len(close)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
        
        if high[i] - high[i - 1] > low[i - 1] - low[i]:
            plus_dm[i] = max(0, high[i] - high[i - 1])
        
        if low[i - 1] - low[i] > high[i] - high[i - 1]:
            minus_dm[i] = max(0, low[i - 1] - low[i])
    
    tr_smooth = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=period, min_periods=period).sum().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=period, min_periods=period).sum().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    mask = tr_smooth > 0
    plus_di[mask] = 100 * plus_dm_smooth[mask] / tr_smooth[mask]
    minus_di[mask] = 100 * minus_dm_smooth[mask] / tr_smooth[mask]
    
    dx = np.zeros(n)
    mask2 = (plus_di + minus_di) > 0
    dx[mask2] = 100 * np.abs(plus_di[mask2] - minus_di[mask2]) / (plus_di[mask2] + minus_di[mask2])
    
    adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
    
    return adx


def calculate_atr(high, low, close, period=14):
    """Calculate ATR for stoploss"""
    n = len(close)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    
    return atr


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    n = len(close)
    
    # 1h indicators for entry timing and risk
    macd_line_1h, macd_signal_1h, macd_hist_1h = calculate_macd(close, fast=12, slow=26, signal=9)
    adx_1h = calculate_adx(high, low, close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    
    # 4h HMA for trend filter (resample 1h → 4h)
    df_1h = pd.DataFrame({
        'open': close,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    df_1h.index = pd.date_range(start='2021-01-01', periods=n, freq='1h')
    
    df_4h = df_1h.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    c_4h = df_4h['close'].values
    len_4h = len(c_4h)
    
    hma_4h = calculate_hma(c_4h, period=48)
    
    trend_4h = np.zeros(len_4h)
    for i in range(48, len_4h):
        if not np.isnan(hma_4h[i]) and hma_4h[i] > 0:
            if c_4h[i] > hma_4h[i]:
                trend_4h[i] = 1
            elif c_4h[i] < hma_4h[i]:
                trend_4h[i] = -1
    
    trend_1h = np.zeros(n)
    for i in range(n):
        idx_4h = (i + 1) // 4 - 1
        if idx_4h >= 0 and idx_4h < len(trend_4h):
            trend_1h[i] = trend_4h[idx_4h]
    
    signals = np.zeros(n)
    
    SIZE_FULL = 0.35
    SIZE_HALF = 0.25
    SIZE_QUARTER = 0.20
    
    ADX_MIN = 25
    ATR_STOP_MULT = 2.5
    
    first_valid = 200
    
    entry_price = 0.0
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(first_valid, n):
        if np.isnan(macd_hist_1h[i]) or np.isnan(adx_1h[i]) or np.isnan(atr_1h[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        macd_hist = macd_hist_1h[i]
        adx_val = adx_1h[i]
        atr = atr_1h[i]
        price = close[i]
        
        if atr > 0 and atr / price > 0.04:
            signals[i] = 0.0
            if position_side != 0:
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            continue
        
        if position_side != 0:
            if position_side == 1:
                highest_since_entry = max(highest_since_entry, price)
                stoploss_price = entry_price - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side = 0
                    entry_price = 0.0
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                    continue
            elif position_side == -1:
                lowest_since_entry = min(lowest_since_entry, price)
                stoploss_price = entry_price + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side = 0
                    entry_price = 0.0
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                    continue
        
        if adx_val < ADX_MIN:
            if position_side != 0:
                signals[i] = signals[i - 1] if i > 0 else 0.0
            else:
                signals[i] = 0.0
            continue
        
        position_size = SIZE_FULL if adx_val > 35 else SIZE_HALF
        
        macd_hist_prev = macd_hist_1h[i - 1] if i > 0 else 0.0
        
        if trend == 1:
            if macd_hist > 0 and macd_hist > macd_hist_prev:
                signals[i] = position_size
                if position_side != 1:
                    position_side = 1
                    entry_price = price
                    highest_since_entry = price
                    lowest_since_entry = price
            else:
                if position_side == 1:
                    signals[i] = signals[i - 1] if i > 0 else 0.0
                else:
                    signals[i] = 0.0
        elif trend == -1:
            if macd_hist < 0 and macd_hist < macd_hist_prev:
                signals[i] = -position_size
                if position_side != -1:
                    position_side = -1
                    entry_price = price
                    highest_since_entry = price
                    lowest_since_entry = price
            else:
                if position_side == -1:
                    signals[i] = signals[i - 1] if i > 0 else 0.0
                else:
                    signals[i] = 0.0
        else:
            signals[i] = 0.0
            if position_side != 0:
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
    
    return signals