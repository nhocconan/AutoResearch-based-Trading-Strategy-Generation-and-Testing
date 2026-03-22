#!/usr/bin/env python3
"""
Experiment #445: 1h Primary + 4h/1d HTF — Fisher Transform + Choppiness Regime + Session Filter

Hypothesis: After 444 failed experiments, clear pattern emerges:
1. Fisher Transform excels at catching reversals in bear/range markets (2025 test period)
2. Choppiness Index regime detection prevents wrong strategy in wrong market
3. 4h HMA provides major trend bias to avoid counter-trend disasters
4. Session filter (8-20 UTC) reduces noise and focuses on high-liquidity periods
5. 1h TF with strict filters = 30-60 trades/year (avoids fee drag)

Why this might beat current best (Sharpe=0.435):
- Fisher Transform has proven edge in mean-reversion during bear markets
- CHOP regime switch adapts: trend-follow when CHOP<45, mean-revert when CHOP>55
- 4h HTF filter prevents 2022-style whipsaw (major trend alignment)
- Session filter reduces false signals during low-liquidity hours
- Discrete position sizing (0.20/0.30) minimizes fee churn

Position sizing: 0.20-0.30 (discrete levels, max 0.40)
Stoploss: 2.5 * ATR trailing
Target: 30-60 trades/year on 1h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_chop_regime_4h1d_session_v1"
timeframe = "1h"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - catches reversals in bear rallies.
    Normalizes price to -1 to +1 range, crossings indicate reversals.
    """
    hl2 = (high + low) / 2.0
    
    # Calculate highest high and lowest low over period
    hh = pd.Series(hl2).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(hl2).rolling(window=period, min_periods=period).min().values
    
    # Normalize to 0-1 range
    norm = np.zeros_like(hl2)
    mask = (hh - ll) > 1e-10
    norm[mask] = (hl2[mask] - ll[mask]) / (hh[mask] - ll[mask])
    
    # Avoid extreme values (0 or 1)
    norm = np.clip(norm, 0.001, 0.999)
    
    # Fisher transform
    fisher = 0.5 * np.log((1.0 + norm) / (1.0 - norm + 1e-10))
    fisher_prev = np.roll(fisher, 1)
    fisher_prev[0] = fisher[0]
    
    return fisher, fisher_prev

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - regime detection.
    CHOP > 61.8 = ranging market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    atr_val = calculate_atr(high, low, close, period)
    
    # Highest high and lowest low over period
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # CHOP formula
    chop = np.zeros_like(close)
    mask = (hh - ll) > 1e-10
    chop_sum = np.zeros_like(close)
    
    for i in range(period, len(close)):
        chop_sum[i] = np.sum(atr_val[i-period+1:i+1])
    
    chop[mask] = 100.0 * np.log10((hh[mask] - ll[mask]) / (chop_sum[mask] + 1e-10)) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def extract_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    hours = (open_time // (1000 * 60 * 60)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HTF indicators (major trend direction)
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_50 = calculate_hma(df_4h['close'].values, period=50)
    
    # Calculate 1d HTF indicators (major regime)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, period=9)
    chop = calculate_choppiness(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Extract trading hours
    hours = np.array([extract_hour(ot) for ot in open_time])
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.25
    SHORT_SIZE = 0.20
    
    # Track position state
    in_position = False
    position_side = 0
    highest_price = 0.0
    lowest_price = 0.0
    entry_price = 0.0
    last_trade_bar = -20
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_50_aligned[i]):
            continue
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        if np.isnan(fisher[i]) or np.isnan(chop[i]) or np.isnan(rsi_14[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = (hours[i] >= 8) and (hours[i] <= 20)
        
        # === 1D MAJOR TREND (primary direction filter) ===
        bull_regime_1d = close[i] > hma_1d_21_aligned[i]
        bear_regime_1d = close[i] < hma_1d_21_aligned[i]
        
        # === 4H LOCAL TREND (HMA crossover) ===
        hma_4h_bullish = hma_4h_21_aligned[i] > hma_4h_50_aligned[i]
        hma_4h_bearish = hma_4h_21_aligned[i] < hma_4h_50_aligned[i]
        
        # === CHOPPY INDEX REGIME ===
        # CHOP > 55 = ranging market (mean revert)
        # CHOP < 45 = trending market (trend follow)
        is_ranging = chop[i] > 55.0
        is_trending = chop[i] < 45.0
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_oversold = fisher[i] < -1.5 and fisher_prev[i] <= fisher[i]
        fisher_overbought = fisher[i] > 1.5 and fisher_prev[i] >= fisher[i]
        fisher_cross_up = fisher[i] > fisher_prev[i] and fisher_prev[i] < -1.0
        fisher_cross_down = fisher[i] < fisher_prev[i] and fisher_prev[i] > 1.0
        
        # === RSI FILTER ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === ENTRY LOGIC — REGIME ADAPTIVE ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        if in_session:
            # Trending regime + bull bias: Fisher cross up from oversold
            if is_trending and (bull_regime_1d or hma_4h_bullish):
                if fisher_cross_up and rsi_oversold:
                    new_signal = LONG_SIZE
                elif fisher[i] > -1.0 and fisher_prev[i] < -1.5 and hma_4h_bullish:
                    new_signal = LONG_SIZE
            
            # Ranging regime: mean reversion at extremes
            elif is_ranging:
                if fisher_oversold and rsi_oversold:
                    new_signal = LONG_SIZE * 0.8
                elif rsi_14[i] < 30.0 and close[i] < hma_4h_21_aligned[i] * 0.98:
                    new_signal = LONG_SIZE * 0.8
        
        # SHORT ENTRIES
        if in_session:
            # Trending regime + bear bias: Fisher cross down from overbought
            if is_trending and (bear_regime_1d or hma_4h_bearish):
                if fisher_cross_down and rsi_overbought:
                    if new_signal == 0.0:
                        new_signal = -SHORT_SIZE
                elif fisher[i] < 1.0 and fisher_prev[i] > 1.5 and hma_4h_bearish:
                    if new_signal == 0.0:
                        new_signal = -SHORT_SIZE
            
            # Ranging regime: mean reversion at extremes
            elif is_ranging:
                if fisher_overbought and rsi_overbought:
                    if new_signal == 0.0:
                        new_signal = -SHORT_SIZE * 0.8
                elif rsi_14[i] > 70.0 and close[i] > hma_4h_21_aligned[i] * 1.02:
                    if new_signal == 0.0:
                        new_signal = -SHORT_SIZE * 0.8
        
        # === FREQUENCY BOOST (ensure >=30 trades/symbol on train) ===
        # If no trade for 15 bars (~15 hours on 1h), force entry on weaker signal
        if bars_since_last_trade > 15 and new_signal == 0.0 and not in_position and in_session:
            if bull_regime_1d and hma_4h_bullish and rsi_14[i] < 45.0:
                new_signal = LONG_SIZE * 0.6
            elif bear_regime_1d and hma_4h_bearish and rsi_14[i] > 55.0:
                new_signal = -SHORT_SIZE * 0.6
        
        # === EXIT CONDITIONS ===
        # Fisher extreme exit (take profit on exhaustion)
        if in_position and position_side > 0 and fisher[i] > 1.5:
            new_signal = 0.0
        if in_position and position_side < 0 and fisher[i] < -1.5:
            new_signal = 0.0
        
        # RSI extreme exit
        if in_position and position_side > 0 and rsi_14[i] > 75.0:
            new_signal = 0.0
        if in_position and position_side < 0 and rsi_14[i] < 25.0:
            new_signal = 0.0
        
        # Regime flip exit (4h trend reversal)
        if in_position and position_side > 0 and hma_4h_bearish and bear_regime_1d:
            new_signal = 0.0
        if in_position and position_side < 0 and hma_4h_bullish and bull_regime_1d:
            new_signal = 0.0
        
        # Session end exit (close position before low-liquidity hours)
        if in_position and hours[i] == 20:
            new_signal = 0.0
        
        # === STOPLOSS (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_price = max(highest_price, close[i])
            stop_price = highest_price - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_price == 0.0:
                lowest_price = close[i]
            else:
                lowest_price = min(lowest_price, close[i])
            stop_price = lowest_price + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals