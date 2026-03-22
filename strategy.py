#!/usr/bin/env python3
"""
Experiment #108: 1d Choppiness Regime + Weekly HMA Trend + RSI Entries + ATR Stop

Hypothesis: Pure trend strategies fail in bear/range markets (2022 crash, 2025 bear).
This strategy adapts to market regime using Choppiness Index:
- CHOP > 61.8 = range market → mean reversion (RSI extremes)
- CHOP < 38.2 = trending market → trend following (breakout)
- 1w HMA provides higher timeframe trend bias
- ATR trailing stop (2.5x) protects against adverse moves

Why this might beat #100 (Sharpe=0.436):
- Regime adaptation works in both bull AND bear markets
- Mean reversion in ranges captures 2022-2025 behavior better
- Weekly HMA filter prevents counter-trend trades in major moves
- Daily timeframe = fewer trades = less fee drag
- Designed to generate trades on ALL symbols (BTC/ETH/SOL)

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete levels
Stoploss: 2.5 * ATR(14)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_chop_regime_1w_hma_rsi_atr_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range/sideways market
    CHOP < 38.2 = trending market
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    atr_vals = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        if np.isnan(atr_vals[i-period+1:i+1]).any():
            continue
        
        sum_atr = np.nansum(atr_vals[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and sum_atr > 0:
            chop[i] = 100 * np.log10(sum_atr / price_range) / np.log10(period)
    
    return chop

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.zeros(len(close))
    mask = avg_loss > 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rs[~mask] = 100
    
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = np.nan
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

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
    chop = calculate_choppiness(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.20
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
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME TREND BIAS ===
        # 1w HMA = major trend direction
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === CHOPPINESS REGIME DETECTION ===
        # CHOP > 61.8 = range market (mean reversion)
        # CHOP < 38.2 = trending market (breakout)
        is_range = chop[i] > 61.8
        is_trend = chop[i] < 38.2
        
        new_signal = 0.0
        
        # === RANGE REGIME: MEAN REVERSION ===
        if is_range:
            # Long: RSI oversold + price above weekly HMA (bullish bias)
            if rsi[i] < 25 and bull_trend_1w:
                new_signal = SIZE_STRONG
            elif rsi[i] < 20:  # Extreme oversold regardless of trend
                new_signal = SIZE_BASE
            
            # Short: RSI overbought + price below weekly HMA (bearish bias)
            if rsi[i] > 75 and bear_trend_1w:
                new_signal = -SIZE_STRONG
            elif rsi[i] > 80:  # Extreme overbought regardless of trend
                new_signal = -SIZE_BASE
        
        # === TREND REGIME: BREAKOUT ===
        elif is_trend:
            # Long breakout: price breaks Donchian upper + weekly bullish
            if close[i] > donchian_upper[i-1] and bull_trend_1w:
                new_signal = SIZE_STRONG
            elif close[i] > donchian_upper[i-1]:
                new_signal = SIZE_BASE
            
            # Short breakout: price breaks Donchian lower + weekly bearish
            if close[i] < donchian_lower[i-1] and bear_trend_1w:
                new_signal = -SIZE_STRONG
            elif close[i] < donchian_lower[i-1]:
                new_signal = -SIZE_BASE
        
        # === NEUTRAL REGIME (38.2 <= CHOP <= 61.8) ===
        # Use RSI extremes only with weekly trend filter
        else:
            if rsi[i] < 20 and bull_trend_1w:
                new_signal = SIZE_BASE
            elif rsi[i] > 80 and bear_trend_1w:
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR ===
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