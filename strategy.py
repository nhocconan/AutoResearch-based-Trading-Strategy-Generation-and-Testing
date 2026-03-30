#!/usr/bin/env python3
"""
Experiment #021: 4h HMA(16) + Donchian(20) + Volume + RSI + ATR Stop

HYPOTHESIS: Combine proven momentum (HMA) with structural breakout (Donchian)
and add RSI filter to avoid overbought/oversold entries, plus volume confirmation.
This creates a confluence of 3+ factors that should:
1. Generate 75-150 total trades (moderate frequency)
2. Have better entry quality than pure momentum strategies
3. Work in both bull (buy breakouts) and bear (short breakups in downtrend)

WHY 4h: Captures multi-day trends without overtrading (unlike 15m/30m).
4h has proven best performer in DB (41% keep rate).

WHY IT SHOULD WORK: HMA gives smooth trend, Donchian gives objective breakout
levels, RSI avoids entries at extremes, volume confirms institutional interest.

KEY: 2-bar minimum hold prevents whipsaw from brief price crossings.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_donchian_rsi_vol_atr_v1"
timeframe = "4h"
leverage = 1.0


def calculate_hma(data, period):
    """Hull Moving Average"""
    n = len(data)
    if n < period:
        return np.full(n, np.nan)
    
    # Convert to pandas for rolling operations
    series = pd.Series(data)
    half = int(np.floor(period / 2))
    sqrt_n = int(np.floor(np.sqrt(period)))
    
    # WMA of half period
    wma_half = series.rolling(window=half, min_periods=half).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)), raw=True
    )
    
    # WMA of full period
    wma_full = series.rolling(window=period, min_periods=period).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)), raw=True
    )
    
    # HMA = WMA(2*WMA_half - WMA_full) of sqrt period
    hma_raw = 2 * wma_half - wma_full
    hma = hma_raw.rolling(window=sqrt_n, min_periods=sqrt_n).apply(
        lambda x: np.sum(x * np.arange(1, len(x) + 1)) / np.sum(np.arange(1, len(x) + 1)), raw=True
    )
    
    return hma.values


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


def calculate_rsi(prices, period=14):
    """Relative Strength Index"""
    n = len(prices)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(prices, prepend=prices[0])
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gains).ewm(alpha=1/period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(alpha=1/period, min_periods=period, adjust=False).mean().values
    
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA for trend (confluence with HMA)
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Local 4h indicators ===
    hma_16 = calculate_hma(close, 16)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Donchian channels (20 periods)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.25  # Conservative sizing
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_bar = 0
    
    warmup = 150  # Need enough for all indicators
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 0:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(hma_16[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d EMA) ===
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        hma_bullish = close[i] > hma_16[i]
        hma_bearish = close[i] < hma_16[i]
        
        # === MOMENTUM CONFIRMATION (RSI) ===
        rsi_neutral_long = 40 < rsi_14[i] < 70  # Not overbought, not oversold
        rsi_neutral_short = 30 < rsi_14[i] < 60  # Not oversold, not overbought
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT ===
        bullish_breakout = high[i] > donchian_high[i - 1]  # Broke yesterday's 20-bar high
        bearish_breakout = low[i] < donchian_low[i - 1]     # Broke yesterday's 20-bar low
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Donchian breakout + volume + trend alignment ===
            if bullish_breakout and vol_spike and price_above_1d_ema and rsi_neutral_long:
                desired_signal = SIZE
            # Backup: strong HMA trend + volume without breakout
            elif hma_bullish and price_above_1d_ema and vol_spike and rsi_neutral_long:
                # Only if close is very close to HMA (within 0.5 ATR)
                if abs(close[i] - hma_16[i]) < 0.5 * atr_14[i]:
                    desired_signal = SIZE
            
            # === SHORT: Donchian breakdown + volume + trend alignment ===
            if bearish_breakout and vol_spike and not price_above_1d_ema and rsi_neutral_short:
                desired_signal = -SIZE
            # Backup: strong HMA downtrend + volume without breakdown
            elif hma_bearish and not price_above_1d_ema and vol_spike and rsi_neutral_short:
                if abs(close[i] - hma_16[i]) < 0.5 * atr_14[i]:
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR from entry) ===
        if in_position:
            bars_held = i - entry_bar
            
            # Minimum hold: 2 bars to avoid whipsaw
            if bars_held >= 2:
                # ATR trailing stop
                if position_side > 0:
                    # Trailing stop rises with price
                    trailing_stop = close[i] - 2.5 * atr_14[i]
                    # Check if stop hit
                    if low[i] < trailing_stop:
                        desired_signal = 0.0
                
                elif position_side < 0:
                    trailing_stop = close[i] + 2.5 * atr_14[i]
                    if high[i] > trailing_stop:
                        desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals