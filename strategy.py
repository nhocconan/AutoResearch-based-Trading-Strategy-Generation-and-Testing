#!/usr/bin/env python3
"""
Experiment #017: 15m Primary + 4h/12h HTF — HMA Trend + RSI Pullback + Volume Confirmation

Hypothesis: After 16 failed experiments, the pattern shows:
- 15m strategies fail when entry conditions are TOO STRICT (0 trades = Sharpe=0)
- Session filters kill trade frequency on crypto (24/7 market)
- CRSI+Choppiness combinations failed 5+ times already
- SUCCESS PATTERN: 4h HMA trend + 15m RSI pullback (from baseline mtf_hma_rsi_zscore)

NEW APPROACH for 15m:
- 4h HMA(21) for major trend bias (proven edge)
- 12h HMA(50) for regime filter (bull/bear)
- 15m RSI(7) for fast entry timing (not RSI14 - too slow for 15m)
- Volume confirmation: taker_buy_volume ratio > 0.55 for longs
- LOOSE entry filters to ensure 50-100 trades/year
- Size: 0.22 (conservative for 15m frequency)
- Stoploss: 2.5x ATR trailing

Why this should work:
- 4h trend filter reduces 15m whipsaws (proven in baseline)
- RSI(7) is faster than RSI(14) for 15m entries
- Volume confirmation adds edge without killing frequency
- Loose RSI thresholds (25-75) ensure trades generate
- Discrete signal sizes minimize fee churn

Target: Sharpe>0.019 (beat current best), DD>-40%, trades>=40 on train, trades>=5 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_rsi_volume_4h12h_v1"
timeframe = "15m"
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
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_ratio(taker_buy_volume, volume):
    """Taker buy volume ratio - measures buying pressure"""
    n = len(volume)
    ratio = np.zeros(n)
    ratio[:] = np.nan
    
    for i in range(n):
        if volume[i] > 1e-10:
            ratio[i] = taker_buy_volume[i] / volume[i]
        else:
            ratio[i] = 0.5
    
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h HMA for regime filter
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=50)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (15m) indicators
    hma_15m = calculate_hma(close, period=13)
    rsi = calculate_rsi(close, period=7)  # Fast RSI for 15m
    atr = calculate_atr(high, low, close, period=14)
    vol_ratio = calculate_volume_ratio(taker_buy_volume, volume)
    
    signals = np.zeros(n)
    SIZE = 0.22  # 22% position size (conservative for 15m)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_15m[i]) or np.isnan(rsi[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (4h HMA) ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === REGIME FILTER (12h HMA) ===
        regime_bull = close[i] > hma_12h_aligned[i]
        regime_bear = close[i] < hma_12h_aligned[i]
        
        # === 15m HMA TREND ===
        hma_bull = close[i] > hma_15m[i]
        hma_bear = close[i] < hma_15m[i]
        
        # === RSI SIGNALS (LOOSE - ensure trades generate) ===
        rsi_oversold = rsi[i] < 45.0  # Not too extreme
        rsi_overbought = rsi[i] > 55.0  # Not too extreme
        rsi_neutral = rsi[i] >= 35.0 and rsi[i] <= 65.0
        
        # === VOLUME CONFIRMATION ===
        vol_buy_pressure = vol_ratio[i] > 0.52  # Slight buy bias
        vol_sell_pressure = vol_ratio[i] < 0.48  # Slight sell bias
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG: 4h bull + 15m pullback + volume confirmation
        # Entry when RSI dips in uptrend (pullback entry)
        if htf_bull and regime_bull and hma_bull:
            if rsi_oversold and vol_buy_pressure:
                desired_signal = SIZE
            elif rsi[i] < 40.0:  # Deeper pullback
                desired_signal = SIZE * 0.8
        
        # SHORT: 4h bear + 15m rally + volume confirmation
        # Entry when RSI rises in downtrend (retracement entry)
        elif htf_bear and regime_bear and hma_bear:
            if rsi_overbought and vol_sell_pressure:
                desired_signal = -SIZE
            elif rsi[i] > 60.0:  # Stronger rally
                desired_signal = -SIZE * 0.8
        
        # Fallback: Strong RSI extremes override HTF (catch reversals)
        if desired_signal == 0.0:
            if rsi[i] < 25.0 and htf_bull:  # Deep oversold in bull
                desired_signal = SIZE * 0.6
            elif rsi[i] > 75.0 and htf_bear:  # Deep overbought in bear
                desired_signal = -SIZE * 0.6
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.5
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.5
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals