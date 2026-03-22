#!/usr/bin/env python3
"""
Experiment #216: 1d HMA Trend + 1w HTF Bias + RSI Momentum + ADX Filter + ATR Stop

Hypothesis: Daily timeframe captures major crypto trends while filtering intraday noise.
Using 1w HMA as higher-timeframe bias prevents counter-trend trades during major regime
shifts. HMA(21) crossover provides entry signals, RSI(14) confirms momentum, ADX(14)>20
filters choppy markets, and ATR(14) trailing stop protects against reversals.

Why 1d might work better than recent failures:
- 1d bars = 1 per day, captures sustained trends without 4h/12h noise
- 1w HMA filter prevents trading against weekly trend (major improvement)
- HMA has less lag than EMA, catches trends earlier
- RSI momentum filter avoids entering at extremes
- ADX > 20 ensures we only trade when trend has strength
- Conservative sizing (0.30) controls drawdown in 2022-style crashes

Learning from failures:
- #204 (1d Donchian): Sharpe=-0.136 - breakout alone insufficient
- #210 (1d KAMA): Sharpe=-0.169 - KAMA too slow for 1d
- #214 (4h DEMA): Sharpe=0.191 - close but needs better HTF filter
- #215 (12h KAMA): Sharpe=0.033 - almost breakeven, needs momentum filter
- Mean reversion fails on crypto (see #207 Sharpe=-9.084)
- Complex regime strategies fail (#209, #212) - keep it simple

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_1w_bias_rsi_adx_atr_v1"
timeframe = "1d"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    
    # Calculate True Range and Directional Movement
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    # Smooth with Wilder's method (EMA with span=period)
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Avoid division by zero
    tr_s = np.where(tr_s == 0, 1e-10, tr_s)
    
    plus_di = 100 * plus_dm_s / tr_s
    minus_di = 100 * minus_dm_s / tr_s
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    ema_21 = calculate_ema(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    prev_hma_21 = 0.0
    prev_hma_50 = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(rsi[i]) or np.isnan(hma_21[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 1w HMA = higher timeframe trend bias (regime filter)
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === TREND STRENGTH FILTER ===
        # ADX > 20 = trending market (not choppy)
        trend_strength = adx[i] > 20
        
        # === HMA CROSSOVER SIGNAL ===
        # HMA21 crossing above HMA50 = bullish crossover
        # HMA21 crossing below HMA50 = bearish crossover
        hma_cross_long = (hma_21[i] > hma_50[i]) and (prev_hma_21 <= prev_hma_50)
        hma_cross_short = (hma_21[i] < hma_50[i]) and (prev_hma_21 >= prev_hma_50)
        
        # === HMA POSITION ===
        # Price above HMA21 = bullish
        # Price below HMA21 = bearish
        hma_bullish = close[i] > hma_21[i]
        hma_bearish = close[i] < hma_21[i]
        
        # === RSI MOMENTUM ===
        # RSI > 50 = bullish momentum (but not overbought > 80)
        # RSI < 50 = bearish momentum (but not oversold < 20)
        rsi_bullish = 50 < rsi[i] < 80
        rsi_bearish = 20 < rsi[i] < 50
        
        # === EMA CONFIRMATION ===
        # Price above EMA21 = short-term bullish
        # Price below EMA21 = short-term bearish
        ema_bullish = close[i] > ema_21[i]
        ema_bearish = close[i] < ema_21[i]
        
        new_signal = 0.0
        
        # === ENTRY CONDITIONS ===
        # Long: 1w bullish + ADX trending + (HMA crossover OR HMA bullish) + RSI/EMA confirmation
        if bull_trend_1w and trend_strength:
            if (hma_cross_long or (hma_bullish and ema_bullish)):
                if rsi_bullish:
                    new_signal = SIZE_BASE
        
        # Short: 1w bearish + ADX trending + (HMA crossover OR HMA bearish) + RSI/EMA confirmation
        if bear_trend_1w and trend_strength:
            if (hma_cross_short or (hma_bearish and ema_bearish)):
                if rsi_bearish:
                    new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.5 * ATR below highest close
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.5 * ATR above lowest close
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === EXIT ON TREND REVERSAL ===
        # Exit long if 1w trend turns bearish
        if in_position and position_side > 0 and bear_trend_1w:
            new_signal = 0.0
        
        # Exit short if 1w trend turns bullish
        if in_position and position_side < 0 and bull_trend_1w:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
        
        # Store previous values for crossover detection
        prev_hma_21 = hma_21[i]
        prev_hma_50 = hma_50[i]
    
    return signals