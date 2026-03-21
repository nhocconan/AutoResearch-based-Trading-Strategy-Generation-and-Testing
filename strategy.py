#!/usr/bin/env python3
"""
Experiment #402: 1d HMA Trend + Weekly HMA Bias + Bollinger Regime + RSI Momentum + ATR Stop
Hypothesis: HMA (Hull Moving Average) provides smoother trend following with less lag than EMA/KAMA.
Weekly HMA gives long-term bias. Bollinger Band Width detects volatility regimes: narrow BB = squeeze
(breakout likely), wide BB = trending (continue trend). RSI(14) with WIDE ranges (25-75) ensures
trade frequency on daily timeframe. Simpler entry logic than #396 (fewer AND conditions) to avoid
0-trade problem. Position size 0.30 discrete with 3*ATR trailing stop for daily timeframe.
Key insight: #396 had positive return but negative Sharpe - entry logic ok, exit timing poor.
This version simplifies entries, uses proven HMA (from best strategy mtf_12h_supertrend_daily_hma_rsi_pullback_v2),
and adds BB regime filter for better timing. Target: Beat Sharpe=0.499.
Timeframe: 1d (REQUIRED), HTF: 1w via mtf_data helper (call ONCE before loop).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_weekly_bollinger_regime_rsi_atr_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    # Band Width = (Upper - Lower) / SMA * 100
    bw = np.zeros(len(close))
    bw[:] = np.nan
    mask = sma > 0
    bw[mask] = (upper[mask] - lower[mask]) / sma[mask] * 100
    return upper, lower, bw, sma

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

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
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    bb_upper, bb_lower, bb_width, bb_sma = calculate_bollinger(close, 20, 2.0)
    rsi = calculate_rsi(close, 14)
    
    # Calculate BB Width percentile for regime (rolling 100 days)
    bb_width_pct = pd.Series(bb_width).rolling(window=100, min_periods=50).apply(
        lambda x: np.sum(x[:-1] < x[-1]) / len(x[:-1]) * 100 if len(x) > 1 else 50
    ).values
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):  # Start after 100 bars for indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(hma_21[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_50[i]) or np.isnan(bb_width[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Weekly trend bias (long-term direction)
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # HMA trend on daily
        hma_bullish = hma_21[i] > hma_50[i]
        hma_bearish = hma_21[i] < hma_50[i]
        
        # Price vs HMA21
        price_above_hma = close[i] > hma_21[i]
        price_below_hma = close[i] < hma_21[i]
        
        # HMA slope
        hma_slope_up = hma_21[i] > hma_21[i-1] if i > 0 else False
        hma_slope_down = hma_21[i] < hma_21[i-1] if i > 0 else False
        
        # Bollinger Band regime
        bb_squeeze = bb_width_pct[i] < 30  # Low volatility = breakout coming
        bb_expanding = bb_width_pct[i] > 70  # High volatility = trending
        
        # RSI momentum (WIDE ranges to ensure trade frequency)
        rsi_ok_long = rsi[i] > 30 and rsi[i] < 80
        rsi_ok_short = rsi[i] > 20 and rsi[i] < 70
        
        # RSI momentum confirmation
        rsi_momentum_long = rsi[i] > 45
        rsi_momentum_short = rsi[i] < 55
        
        new_signal = 0.0
        
        # === LONG ENTRIES (simplified logic for trade frequency) ===
        # Primary: HMA bullish + Weekly bullish + Price above HMA + RSI ok
        if hma_bullish and weekly_bullish and price_above_hma and rsi_ok_long:
            new_signal = SIZE_ENTRY
        # Secondary: HMA bullish + HMA slope up + RSI momentum (weekly neutral)
        elif hma_bullish and hma_slope_up and rsi_momentum_long:
            new_signal = SIZE_ENTRY
        # Tertiary: Price crosses above HMA21 + RSI confirmation
        elif price_above_hma and close[i-1] <= hma_21[i-1] and rsi[i] > 40:
            new_signal = SIZE_ENTRY
        # Quaternary: BB squeeze breakout long + Weekly bullish
        elif bb_squeeze and price_above_hma and weekly_bullish:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (simplified logic for trade frequency) ===
        # Primary: HMA bearish + Weekly bearish + Price below HMA + RSI ok
        if hma_bearish and weekly_bearish and price_below_hma and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        # Secondary: HMA bearish + HMA slope down + RSI momentum (weekly neutral)
        elif hma_bearish and hma_slope_down and rsi_momentum_short:
            new_signal = -SIZE_ENTRY
        # Tertiary: Price crosses below HMA21 + RSI confirmation
        elif price_below_hma and close[i-1] >= hma_21[i-1] and rsi[i] < 60:
            new_signal = -SIZE_ENTRY
        # Quaternary: BB squeeze breakout short + Weekly bearish
        elif bb_squeeze and price_below_hma and weekly_bearish:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) - 3*ATR for daily timeframe ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (3*ATR from highest for daily timeframe)
            current_stop = highest_close - 3.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 3.0 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (3*ATR from lowest for daily timeframe)
            current_stop = lowest_close + 3.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 3.0 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 3.0 * atr[i] if position_side > 0 else close[i] + 3.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 3.0 * atr[i] if position_side > 0 else close[i] + 3.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals