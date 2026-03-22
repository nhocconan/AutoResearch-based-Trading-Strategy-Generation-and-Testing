#!/usr/bin/env python3
"""
Experiment #098: 30m EMA Crossover + 4h HMA Trend + Volume Confirmation + ATR Stop
Hypothesis: 30m timeframe captures medium-term swings with less noise than 15m.
4h HMA provides stable trend bias (proven in #088 with Sharpe=0.223).
EMA(8/21) crossover on 30m catches trend changes faster than Supertrend.
Volume confirmation filters false breakouts (common on 30m).
RSI momentum (not extremes) ensures we enter with momentum, not against it.
ATR trailing stop protects from reversals.

Why this might work on 30m (learning from #086 Sharpe=-0.616):
- #086 used Supertrend which whipsawed too much on 30m
- EMA crossover is smoother and generates more consistent signals
- Volume filter reduces false entries (critical on 30m)
- Looser entry conditions ensure trades on ALL symbols (BTC, ETH, SOL)
- Conservative position sizing (0.25 base) controls drawdown

Timeframe: 30m (REQUIRED), HTF: 4h via mtf_data helper (call ONCE before loop).
Position sizing: 0.25 base, 0.30 strong signals. Stoploss at 2.0*ATR.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_ema_crossover_4h_hma_volume_rsi_v1"
timeframe = "30m"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[~mask] = 100.0
    
    return rsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_volume_sma(volume, period=20):
    """Calculate volume SMA for volume confirmation."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    ema_8 = calculate_ema(close, 8)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    vol_sma = calculate_volume_sma(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_8[i]) or np.isnan(ema_21[i]) or np.isnan(ema_50[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_sma[i]) or vol_sma[i] == 0:
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = higher timeframe trend bias (proven in #088)
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === EMA CROSSOVER SIGNAL ===
        # Fast EMA (8) crossing above/below slow EMA (21)
        ema_crossover_long = ema_8[i] > ema_21[i]
        ema_crossover_short = ema_8[i] < ema_21[i]
        
        # EMA alignment confirmation (8 > 21 > 50 for strong long)
        ema_aligned_long = ema_8[i] > ema_21[i] and ema_21[i] > ema_50[i]
        ema_aligned_short = ema_8[i] < ema_21[i] and ema_21[i] < ema_50[i]
        
        # === VOLUME CONFIRMATION ===
        # Volume must be above average to confirm breakout
        volume_confirmed = volume[i] > 1.0 * vol_sma[i]
        volume_strong = volume[i] > 1.5 * vol_sma[i]
        
        # === RSI MOMENTUM FILTER ===
        # RSI > 50 for longs (momentum), RSI < 50 for shorts
        rsi_momentum_long = rsi[i] > 45
        rsi_momentum_short = rsi[i] < 55
        
        # RSI not at extremes (avoid mean reversion traps)
        rsi_not_overbought = rsi[i] < 75
        rsi_not_oversold = rsi[i] > 25
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS (loose to ensure trades) ===
        # Path 1: EMA aligned + 4h bullish + volume confirmed (strong signal)
        if ema_aligned_long and bull_trend_4h and volume_confirmed:
            if rsi_momentum_long and rsi_not_overbought:
                if volume_strong:
                    new_signal = SIZE_STRONG
                else:
                    new_signal = SIZE_BASE
        
        # Path 2: EMA crossover + 4h bullish (simpler, ensures trades)
        if new_signal == 0.0 and ema_crossover_long and bull_trend_4h:
            if rsi_momentum_long or volume_confirmed:
                new_signal = SIZE_BASE
        
        # Path 3: EMA aligned + 4h bullish only (fallback for all symbols)
        if new_signal == 0.0 and ema_aligned_long and bull_trend_4h:
            new_signal = SIZE_BASE
        
        # Path 4: EMA crossover + volume strong (momentum entry)
        if new_signal == 0.0 and ema_crossover_long and volume_strong:
            if rsi_momentum_long:
                new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS (loose to ensure trades) ===
        # Path 1: EMA aligned + 4h bearish + volume confirmed (strong signal)
        if ema_aligned_short and bear_trend_4h and volume_confirmed:
            if rsi_momentum_short and rsi_not_oversold:
                if volume_strong:
                    new_signal = -SIZE_STRONG
                else:
                    new_signal = -SIZE_BASE
        
        # Path 2: EMA crossover + 4h bearish (simpler, ensures trades)
        if new_signal == 0.0 and ema_crossover_short and bear_trend_4h:
            if rsi_momentum_short or volume_confirmed:
                new_signal = -SIZE_BASE
        
        # Path 3: EMA aligned + 4h bearish only (fallback for all symbols)
        if new_signal == 0.0 and ema_aligned_short and bear_trend_4h:
            new_signal = -SIZE_BASE
        
        # Path 4: EMA crossover + volume strong (momentum entry)
        if new_signal == 0.0 and ema_crossover_short and volume_strong:
            if rsi_momentum_short:
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR ===
        # Update trailing highs/lows for active positions
        if in_position and position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            # Trailing stop: 2.0 * ATR below highest close
            stoploss_price = highest_close - 2.0 * atr[i]
            if close[i] < stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        if in_position and position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            # Trailing stop: 2.0 * ATR above lowest close
            stoploss_price = lowest_close + 2.0 * atr[i]
            if close[i] > stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        # Update position tracking
        # Entering new position
        if new_signal != 0.0 and not in_position:
            in_position = True
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Reversing position
        elif new_signal != 0.0 and in_position and np.sign(new_signal) != position_side:
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Exiting position
        elif new_signal == 0.0 and in_position:
            in_position = False
            position_side = 0
            entry_price = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals