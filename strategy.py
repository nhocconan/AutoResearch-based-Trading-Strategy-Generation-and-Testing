#!/usr/bin/env python3
"""
Experiment #156: 1d Regime-Adaptive Strategy with Weekly Trend Filter

Hypothesis: Daily timeframe captures major trend moves while weekly HMA provides
stable trend bias. Using Bollinger Band Width to detect volatility regime allows
switching between trend-following (low vol/expanding) and mean-reversion (high vol/contracting).
This adapts to both bull markets (2021) and bear/range markets (2022, 2025).

Why 1d might work:
- Fewer trades = lower fee drag (critical after seeing 15m/30m failures)
- Captures major moves without whipsaw noise
- Weekly filter prevents counter-trend trades in strong trends
- Regime detection adapts to market conditions (trend vs range)

Key innovations vs failed strategies:
- Regime-adaptive entry logic (not pure trend or pure mean-revert)
- Asymmetric position sizing (larger when trend + regime agree)
- Loose enough conditions to ensure ≥10 trades (learned from #142, #143)
- ATR trailing stop for risk management

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 base, 0.35 strong confluence
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_regime_adaptive_1w_hma_bb_atr_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = np.nan
    return rsi

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    bb_width = (upper - lower) / sma
    return upper, lower, sma, bb_width

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = len(close)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        elif minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    atr = calculate_atr(high, low, close, period)
    
    plus_di = pd.Series(100 * plus_dm / np.where(atr == 0, 1, atr)).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(100 * minus_dm / np.where(atr == 0, 1, atr)).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, 1, (plus_di + minus_di))
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx, plus_di, minus_di

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

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
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_sma, bb_width = calculate_bollinger_bands(close, 20, 2.0)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    
    # Calculate BB width percentile for regime detection
    bb_width_percentile = pd.Series(bb_width).rolling(window=100, min_periods=50).apply(
        lambda x: np.percentile(x[~np.isnan(x)], 50) if len(x[~np.isnan(x)]) > 0 else np.nan
    ).values
    bb_width_current_percentile = pd.Series(bb_width).rolling(window=100, min_periods=50).apply(
        lambda x: np.searchsorted(np.sort(x[~np.isnan(x)]), x.iloc[-1]) / len(x[~np.isnan(x)]) if len(x[~np.isnan(x)]) > 0 else np.nan
    ).values if len(bb_width) > 50 else np.zeros(n)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    
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
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_width[i]) or np.isnan(bb_sma[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 1w HMA = higher timeframe trend bias
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION ===
        # Low ADX = ranging market, High ADX = trending market
        is_trending = adx[i] > 25
        is_ranging = adx[i] < 20
        
        # BB width expansion/contraction
        bb_expanding = bb_width[i] > bb_width[i-5] if i >= 5 and not np.isnan(bb_width[i-5]) else False
        bb_contracting = bb_width[i] < bb_width[i-5] if i >= 5 and not np.isnan(bb_width[i-5]) else False
        
        # === ENTRY CONDITIONS ===
        new_signal = 0.0
        
        # LONG entries - multiple pathways to ensure trade frequency
        long_score = 0
        
        # Pathway 1: Trend following (1w bullish + ADX trending + price above BB mid)
        if bull_trend_1w and is_trending and close[i] > bb_sma[i]:
            long_score += 2
        
        # Pathway 2: Mean reversion in range (1w bullish + ADX ranging + RSI oversold)
        if bull_trend_1w and is_ranging and rsi[i] < 35:
            long_score += 2
        
        # Pathway 3: Pullback to support (1w bullish + price near BB lower + RSI < 45)
        if bull_trend_1w and close[i] < bb_lower[i] * 1.02 and rsi[i] < 45:
            long_score += 2
        
        # Pathway 4: BB squeeze breakout (1w bullish + BB contracting + price breaks above BB mid)
        if bull_trend_1w and bb_contracting and close[i] > bb_sma[i] and close[i-1] <= bb_sma[i-1]:
            long_score += 2
        
        # SHORT entries - multiple pathways
        short_score = 0
        
        # Pathway 1: Trend following (1w bearish + ADX trending + price below BB mid)
        if bear_trend_1w and is_trending and close[i] < bb_sma[i]:
            short_score += 2
        
        # Pathway 2: Mean reversion in range (1w bearish + ADX ranging + RSI overbought)
        if bear_trend_1w and is_ranging and rsi[i] > 65:
            short_score += 2
        
        # Pathway 3: Rally to resistance (1w bearish + price near BB upper + RSI > 55)
        if bear_trend_1w and close[i] > bb_upper[i] * 0.98 and rsi[i] > 55:
            short_score += 2
        
        # Pathway 4: BB squeeze breakdown (1w bearish + BB contracting + price breaks below BB mid)
        if bear_trend_1w and bb_contracting and close[i] < bb_sma[i] and close[i-1] >= bb_sma[i-1]:
            short_score += 2
        
        # Generate signal based on score
        if long_score >= 2:
            if long_score >= 4:
                new_signal = SIZE_STRONG
            else:
                new_signal = SIZE_BASE
        
        if short_score >= 2:
            if short_score >= 4:
                new_signal = -SIZE_STRONG
            else:
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Update trailing highs/lows for active positions
        if in_position and position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            # Trailing stop: 2.5 * ATR below highest close
            stoploss_price = highest_close - 2.5 * atr[i]
            if close[i] < stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        if in_position and position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            # Trailing stop: 2.5 * ATR above lowest close
            stoploss_price = lowest_close + 2.5 * atr[i]
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