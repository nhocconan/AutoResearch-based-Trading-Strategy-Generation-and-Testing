#!/usr/bin/env python3
"""
Experiment #396: 1d KAMA Adaptive Trend + Weekly HMA Bias + Choppiness Regime + RSI Momentum + ATR Stop
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market efficiency - moves fast in trends,
slow in ranges. This should reduce whipsaw vs fixed EMA/HMA. Weekly HMA provides long-term trend bias.
Choppiness Index (CHOP) detects range vs trend regimes: CHOP>61.8=range (avoid trades), CHOP<38.2=trend.
RSI(14) confirms momentum without being too restrictive (30-70 range). ATR(14) stoploss at 2.5x for
daily timeframe (wider stops needed on 1d). Position size 0.30 discrete. Daily timeframe should have
fewer but higher-quality trades. Target: Beat Sharpe=0.499 (current best mtf_12h_supertrend_daily_hma_rsi_pullback_v2).
Key insight: KAMA's adaptive nature should handle 2022 crash and 2025 bear better than fixed MA crossovers.
Timeframe: 1d (REQUIRED for this experiment), HTF: 1w for long-term trend bias via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_weekly_hma_chop_regime_rsi_atr_v1"
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

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise - moves fast in trending markets, slow in ranging.
    Efficiency Ratio (ER) measures trend strength: ER = |change| / sum(|changes|)
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio
    change = np.abs(close - np.roll(close, period))
    change[:period] = np.nan
    
    volatility = np.zeros(n)
    for i in range(period, n):
        volatility[i] = np.sum(np.abs(close[i-period+1:i+1] - np.roll(close[i-period+1:i+1], 1))[1:])
    
    er = np.zeros(n)
    er[:] = np.nan
    mask = volatility > 0
    er[mask] = change[mask] / volatility[mask]
    er = np.clip(er, 0, 1)
    
    # Calculate smoothing constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = np.zeros(n)
    sc[:] = np.nan
    valid_er = ~np.isnan(er)
    sc[valid_er] = (er[valid_er] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[period] = close[period]
    
    # Calculate KAMA
    for i in range(period + 1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean reversion likely)
    CHOP < 38.2 = trending market (trend following likely)
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
        else:
            atr_sum = 0
            for j in range(i-period+1, i+1):
                tr = max(high[j] - low[j], 
                        abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j],
                        abs(low[j] - close[j-1]) if j > 0 else high[j] - low[j])
                atr_sum += tr
            
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

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
    kama = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    chop = calculate_choppiness(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    
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
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(kama[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Weekly trend bias (long-term direction)
        weekly_bullish = close[i] > hma_1w_aligned[i]
        weekly_bearish = close[i] < hma_1w_aligned[i]
        
        # Choppiness regime filter
        is_trending = chop[i] < 55  # Slightly relaxed from 38.2 to get more trades
        is_ranging = chop[i] > 55
        
        # KAMA trend direction
        kama_bullish = close[i] > kama[i]
        kama_bearish = close[i] < kama[i]
        
        # KAMA slope (momentum)
        kama_slope_bullish = kama[i] > kama[i-1] if i > 0 else False
        kama_slope_bearish = kama[i] < kama[i-1] if i > 0 else False
        
        # RSI momentum filter (loose to ensure trade frequency on daily)
        rsi_ok_long = rsi[i] > 35 and rsi[i] < 75
        rsi_ok_short = rsi[i] > 25 and rsi[i] < 65
        
        # RSI extreme for stronger signals
        rsi_strong_long = rsi[i] > 45 and rsi[i] < 70
        rsi_strong_short = rsi[i] > 30 and rsi[i] < 55
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple conditions to ensure trades on daily) ===
        # Primary: KAMA bullish + Weekly bullish + Trending + RSI ok
        if kama_bullish and weekly_bullish and is_trending and rsi_ok_long:
            new_signal = SIZE_ENTRY
        # Secondary: KAMA bullish + KAMA slope up + Weekly bullish + RSI ok
        elif kama_bullish and kama_slope_bullish and weekly_bullish and rsi[i] > 40:
            new_signal = SIZE_ENTRY
        # Tertiary: KAMA bullish + Trending + RSI momentum (weekly neutral ok)
        elif kama_bullish and is_trending and rsi_strong_long:
            new_signal = SIZE_ENTRY
        # Quaternary: Weekly bullish + KAMA bullish + RSI ok (chop neutral)
        elif weekly_bullish and kama_bullish and rsi_ok_long:
            new_signal = SIZE_ENTRY
        # Quintenary: KAMA crossover (price crosses above KAMA) + RSI filter
        elif kama_bullish and close[i-1] <= kama[i-1] and rsi[i] > 40:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple conditions to ensure trades on daily) ===
        # Primary: KAMA bearish + Weekly bearish + Trending + RSI ok
        if kama_bearish and weekly_bearish and is_trending and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        # Secondary: KAMA bearish + KAMA slope down + Weekly bearish + RSI ok
        elif kama_bearish and kama_slope_bearish and weekly_bearish and rsi[i] < 60:
            new_signal = -SIZE_ENTRY
        # Tertiary: KAMA bearish + Trending + RSI momentum (weekly neutral ok)
        elif kama_bearish and is_trending and rsi_strong_short:
            new_signal = -SIZE_ENTRY
        # Quaternary: Weekly bearish + KAMA bearish + RSI ok (chop neutral)
        elif weekly_bearish and kama_bearish and rsi_ok_short:
            new_signal = -SIZE_ENTRY
        # Quintenary: KAMA crossover (price crosses below KAMA) + RSI filter
        elif kama_bearish and close[i-1] >= kama[i-1] and rsi[i] < 60:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from highest for daily timeframe)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from lowest for daily timeframe)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
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
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
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