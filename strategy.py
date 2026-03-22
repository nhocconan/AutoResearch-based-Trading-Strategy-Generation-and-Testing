#!/usr/bin/env python3
"""
Experiment #235: 15m Trend Pullback with 4h HMA + 1h RSI + Choppiness Filter

Hypothesis: 15m timeframe captures intraday momentum while 4h HMA provides stable
trend bias. Entry on RSI pullbacks (RSI<40 in uptrend, RSI>60 in downtrend) reduces
whipsaw compared to breakout entries. Choppiness Index filters range markets where
trend strategies fail. This combines the proven 4h HMA filter from best strategy
with faster 15m entries for better risk/reward.

Why 15m might work:
- 15m = 96 bars/day, captures intraday swings without 5m noise
- 4h HMA = proven trend filter from current best strategy (Sharpe=0.478)
- 1h RSI pullback = enters on dips rather than breakouts (better R:R)
- Choppiness > 61.8 = skip trading (range detection)
- Conservative sizing (0.25) controls drawdown

Learning from failures:
- #223 (15m mean rev): Sharpe=-2.849 - mean reversion fails on crypto
- #229 (15m Donchian): Sharpe=-3.173 - breakouts whipsaw without HTF filter
- Need STRONG HTF filter (4h HMA) to avoid counter-trend trades
- Choppiness filter prevents trading in ranges (where trend strategies die)
- RSI pullback > RSI breakout for entry timing

Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h + 1h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_trend_pullback_4h_hma_1h_rsi_chop_atr_v1"
timeframe = "15m"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range/choppy market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.zeros(n)
    
    # Calculate ATR for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / price_range) / np.log10(period)
        else:
            chop[i] = 50  # neutral
    
    # Fill initial values
    chop[:period] = 50
    
    return chop

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
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    rsi_1h = calculate_rsi(df_1h['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi_15m = calculate_rsi(close, 14)
    chop = calculate_choppiness(high, low, close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_REDUCED = 0.15
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(rsi_1h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]) or np.isnan(rsi_15m[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS (4h HMA) ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === CHOPPINESS FILTER ===
        # CHOP > 61.8 = range market (skip trading)
        # CHOP < 61.8 = trending market (allow trades)
        is_trending = chop[i] < 61.8
        
        # === 1h RSI MOMENTUM FILTER ===
        # In uptrend: want 1h RSI not overbought (>30, <70)
        # In downtrend: want 1h RSI not oversold (>30, <70)
        rsi_1h_neutral = 30 < rsi_1h_aligned[i] < 70
        
        # === 15m RSI PULLBACK ENTRY ===
        # Long: RSI<45 (pullback) in uptrend
        # Short: RSI>55 (rally) in downtrend
        rsi_pullback_long = rsi_15m[i] < 45
        rsi_pullback_short = rsi_15m[i] > 55
        
        # === EMA CONFIRMATION ===
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        new_signal = 0.0
        
        # === ENTRY CONDITIONS ===
        # Long: 4h bullish + trending + RSI pullback + (1h RSI neutral OR EMA bullish)
        if bull_trend_4h and is_trending and rsi_pullback_long:
            if rsi_1h_neutral or ema_bullish:
                new_signal = SIZE_BASE
        
        # Short: 4h bearish + trending + RSI pullback + (1h RSI neutral OR EMA bearish)
        if bear_trend_4h and is_trending and rsi_pullback_short:
            if rsi_1h_neutral or ema_bearish:
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
        
        # === TAKE PROFIT / POSITION REDUCTION ===
        # Reduce position at 2R profit (optional, keeps some exposure)
        if in_position and new_signal != 0.0:
            if position_side > 0 and entry_price > 0:
                profit_pct = (close[i] - entry_price) / entry_price
                if profit_pct > 2 * (2.5 * atr[i] / entry_price):  # 2R profit
                    new_signal = SIZE_REDUCED  # Reduce to half position
            
            if position_side < 0 and entry_price > 0:
                profit_pct = (entry_price - close[i]) / entry_price
                if profit_pct > 2 * (2.5 * atr[i] / entry_price):  # 2R profit
                    new_signal = -SIZE_REDUCED  # Reduce to half position
        
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
            # else: maintaining same position direction (possibly reduced)
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals