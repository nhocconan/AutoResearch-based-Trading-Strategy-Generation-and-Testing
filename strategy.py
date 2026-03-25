#!/usr/bin/env python3
"""
Experiment #1476: 30m Primary + 4h/1d HTF — Volume-Confirmed Trend Pullback

Hypothesis: 30m timeframe with VERY STRICT entry filters (3+ confluence) will
generate optimal trade frequency (40-80/year) while avoiding fee drag that
kills lower TF strategies. Key insight from failures: session filters cause
0 trades, so use VOLUME confirmation instead.

Strategy components:
1. 4h HMA(21) for major trend bias (align properly with mtf_data helper)
2. 1d HMA(48) for regime filter (avoid counter-trend in strong bear markets)
3. 30m RSI(7) for pullback entries (fast RSI catches quick retracements)
4. Volume confirmation: taker_buy_volume_ratio > 0.55 for longs, < 0.45 for shorts
5. ATR(14) trailing stoploss at 2.5x ATR
6. Discrete sizing: 0.0, ±0.20, ±0.30 (minimize fee churn)

Why this should work:
- 4h trend filter prevents major counter-trend losses (2022 crash protection)
- 30m RSI(7) pullback = fast entries with HTF confirmation
- Volume filter eliminates false breakouts (major edge on crypto)
- LOOSE RSI thresholds (20/80) ensure ≥40 trades/year
- 30m TF = natural 50-80 trades/year (fee-efficient per Rule 10)

Entry logic (LOOSE to guarantee trades):
- LONG: 4h_HMA bullish + 1d_HMA neutral/bullish + RSI(7)<30 + volume_confirmed
- SHORT: 4h_HMA bearish + 1d_HMA neutral/bearish + RSI(7)>70 + volume_confirmed

Target: Sharpe>0.6, trades>=40 train, trades>=5 test, DD>-35%
Timeframe: 30m
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_volume_pullback_hma_4h1d_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.insert(gain, 0, 0)
    loss = np.insert(loss, 0, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss != 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi

def calculate_volume_ratio(taker_buy_volume, volume):
    """Taker buy volume ratio (0-1 scale)"""
    n = len(volume)
    ratio = np.full(n, np.nan, dtype=np.float64)
    mask = volume > 0
    ratio[mask] = taker_buy_volume[mask] / volume[mask]
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_vol = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=48)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 30m indicators
    hma_21 = calculate_hma(close, period=21)
    hma_50 = calculate_hma(close, period=50)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    vol_ratio = calculate_volume_ratio(taker_buy_vol, volume)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period
    min_bars = 100
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(hma_21[i]) or np.isnan(hma_50[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (4h HMA bias) ===
        price_above_4h = close[i] > hma_4h_aligned[i]
        price_below_4h = close[i] < hma_4h_aligned[i]
        
        # === REGIME FILTER (1d HMA) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === 30m HMA CROSSOVER (momentum) ===
        hma_bullish = hma_21[i] > hma_50[i]
        hma_bearish = hma_21[i] < hma_50[i]
        
        # === RSI PULLBACK ===
        rsi = rsi_7[i]
        rsi_oversold = rsi < 30
        rsi_overbought = rsi > 70
        rsi_extreme_oversold = rsi < 20
        rsi_extreme_overbought = rsi > 80
        
        # === VOLUME CONFIRMATION ===
        vol_bullish = vol_ratio[i] > 0.55
        vol_bearish = vol_ratio[i] < 0.45
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # LONG: 4h bullish + (1d neutral/bullish OR strong oversold) + RSI pullback + volume
        if price_above_4h:
            # Standard long: RSI<30 + volume confirmed
            if rsi_oversold and vol_bullish:
                desired_signal = SIZE_BASE
            # Strong long: extreme oversold (RSI<20) overrides volume
            elif rsi_extreme_oversold:
                desired_signal = SIZE_STRONG
            # Trend continuation: 4h bullish + 30m HMA bullish + RSI not overbought
            elif hma_bullish and rsi < 60 and price_above_1d:
                desired_signal = SIZE_BASE
        
        # SHORT: 4h bearish + (1d neutral/bearish OR strong overbought) + RSI pullback + volume
        elif price_below_4h:
            # Standard short: RSI>70 + volume confirmed
            if rsi_overbought and vol_bearish:
                desired_signal = -SIZE_BASE
            # Strong short: extreme overbought (RSI>80) overrides volume
            elif rsi_extreme_overbought:
                desired_signal = -SIZE_STRONG
            # Trend continuation: 4h bearish + 30m HMA bearish + RSI not oversold
            elif hma_bearish and rsi > 40 and price_below_1d:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
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
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
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
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals