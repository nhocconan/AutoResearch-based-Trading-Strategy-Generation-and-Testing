#!/usr/bin/env python3
"""
Experiment #291: 6h Primary + 1d/1w HTF — Weekly Pivot Bounce + RSI Divergence v1

Hypothesis: 6h timeframe captures multi-day swings ideal for weekly pivot reactions.
Weekly pivot levels (WS1, WS2, WM1, WM2) act as strong S/R on 6h charts.
Combined with RSI divergence for entry timing and 1d/1w HMA for trend bias.

Key differences from failed #287 (funding_chop_crsi):
1. REMOVED funding rate (didn't help on 6h timeframe)
2. ADDED weekly pivot levels as primary S/R (proven on 6h)
3. ADDED RSI divergence (hidden + regular) for entry confirmation
4. ADDED volume spike filter (1.5x avg) to confirm breakouts
5. SIMPLIFIED regime detection (pivot-based, not CHOP-based)

Weekly Pivot Calculation (Woodie):
P = (H + L + 2*C) / 4
R1 = 2*P - L
S1 = 2*P - H
R2 = P + (H - L)
S2 = P - (H - L)

Entry Logic:
- Long: Price near WS1/WS2 + RSI bullish divergence + 1d HMA bull + volume spike
- Short: Price near WR1/WR2 + RSI bearish divergence + 1d HMA bear + volume spike
- Breakout: Price breaks weekly pivot P + 1w HMA alignment + volume confirm

Position sizing: 0.25 base, 0.30 when 1w aligned (discrete levels)
Stoploss: 2.5x ATR from entry

Target: Sharpe>0.40, DD>-35%, trades>=30 train, trades>=3 test
Timeframe: 6h (30-60 trades/year target)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_weekly_pivot_rsi_div_vol_1d1w_v1"
timeframe = "6h"
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
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_weekly_pivots(high, low, close, lookback_weeks=1):
    """
    Weekly Pivot Levels (Woodie formula)
    P = (H + L + 2*C) / 4
    R1 = 2*P - L
    S1 = 2*P - H
    R2 = P + (H - L)
    S2 = P - (H - L)
    
    Returns arrays for P, R1, S1, R2, S2 aligned to 6h bars
    """
    n = len(close)
    P = np.zeros(n)
    R1 = np.zeros(n)
    S1 = np.zeros(n)
    R2 = np.zeros(n)
    S2 = np.zeros(n)
    P[:] = np.nan
    R1[:] = np.nan
    S1[:] = np.nan
    R2[:] = np.nan
    S2[:] = np.nan
    
    # Weekly bars: approximately 28 x 6h bars per week
    bars_per_week = 28
    
    for i in range(bars_per_week * lookback_weeks, n):
        # Get previous week's H, L, C
        week_start = i - bars_per_week * lookback_weeks
        week_end = i
        
        if week_start >= 0:
            week_high = np.nanmax(high[week_start:week_end])
            week_low = np.nanmin(low[week_start:week_end])
            week_close = close[week_end - 1]
            
            if not np.isnan(week_high) and not np.isnan(week_low) and not np.isnan(week_close):
                pivot = (week_high + week_low + 2.0 * week_close) / 4.0
                P[i] = pivot
                R1[i] = 2.0 * pivot - week_low
                S1[i] = 2.0 * pivot - week_high
                R2[i] = pivot + (week_high - week_low)
                S2[i] = pivot - (week_high - week_low)
    
    return P, R1, S1, R2, S2

def calculate_rsi_divergence(close, rsi, lookback=5):
    """
    Detect RSI divergence (regular and hidden)
    Returns: div_bull (bullish div), div_bear (bearish div)
    
    Regular Bullish: Price lower low, RSI higher low
    Regular Bearish: Price higher high, RSI lower high
    Hidden Bullish: Price higher low, RSI lower low (trend continuation)
    Hidden Bearish: Price lower high, RSI higher high (trend continuation)
    """
    n = len(close)
    div_bull = np.zeros(n, dtype=bool)
    div_bear = np.zeros(n, dtype=bool)
    
    for i in range(lookback * 2, n):
        if np.isnan(rsi[i]) or np.isnan(rsi[i-lookback]):
            continue
        
        # Find local extrema in price and RSI
        price_window = close[i-lookback:i+1]
        rsi_window = rsi[i-lookback:i+1]
        
        if len(price_window) < 3 or len(rsi_window) < 3:
            continue
        
        # Check for bullish divergence (price making lower low, RSI making higher low)
        price_low_idx = np.argmin(price_window)
        rsi_low_idx = np.argmin(rsi_window)
        
        if price_low_idx > 0 and rsi_low_idx > 0:
            # Regular bullish: price LL, RSI HL
            if close[i] < close[i-lookback] and rsi[i] > rsi[i-lookback]:
                div_bull[i] = True
            # Hidden bullish: price HL, RSI LL (in uptrend)
            elif close[i] > close[i-lookback] and rsi[i] < rsi[i-lookback]:
                div_bull[i] = True
        
        # Check for bearish divergence (price making higher high, RSI making lower high)
        price_high_idx = np.argmax(price_window)
        rsi_high_idx = np.argmax(rsi_window)
        
        if close[i] > close[i-lookback] and rsi[i] < rsi[i-lookback]:
            div_bear[i] = True
        elif close[i] < close[i-lookback] and rsi[i] > rsi[i-lookback]:
            div_bear[i] = True
    
    return div_bull, div_bear

def calculate_volume_spike(volume, period=20, threshold=1.5):
    """Detect volume spikes (volume > threshold * avg volume)"""
    n = len(volume)
    spike = np.zeros(n, dtype=bool)
    
    for i in range(period, n):
        avg_vol = np.nanmean(volume[i-period:i])
        if avg_vol > 1e-10 and volume[i] > threshold * avg_vol:
            spike[i] = True
    
    return spike

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (6h) indicators
    hma_6h = calculate_hma(close, period=21)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    sma_200 = calculate_sma(close, 200)
    
    # Weekly pivot levels
    pivot_P, pivot_R1, pivot_S1, pivot_R2, pivot_S2 = calculate_weekly_pivots(high, low, close, lookback_weeks=1)
    
    # RSI divergence detection
    div_bull, div_bear = calculate_rsi_divergence(close, rsi, lookback=5)
    
    # Volume spike detection
    vol_spike = calculate_volume_spike(volume, period=20, threshold=1.5)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_6h[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(pivot_P[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # 1w for major trend (optional boost)
        htf_1w_bull = not np.isnan(hma_1w_aligned[i]) and close[i] > hma_1w_aligned[i]
        htf_1w_bear = not np.isnan(hma_1w_aligned[i]) and close[i] < hma_1w_aligned[i]
        
        # === 6h HMA TREND ===
        hma_bull = close[i] > hma_6h[i]
        hma_bear = close[i] < hma_6h[i]
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === PIVOT PROXIMITY (within 1% of pivot level) ===
        near_S1 = abs(close[i] - pivot_S1[i]) / close[i] < 0.01 if not np.isnan(pivot_S1[i]) else False
        near_S2 = abs(close[i] - pivot_S2[i]) / close[i] < 0.01 if not np.isnan(pivot_S2[i]) else False
        near_R1 = abs(close[i] - pivot_R1[i]) / close[i] < 0.01 if not np.isnan(pivot_R1[i]) else False
        near_R2 = abs(close[i] - pivot_R2[i]) / close[i] < 0.01 if not np.isnan(pivot_R2[i]) else False
        near_P = abs(close[i] - pivot_P[i]) / close[i] < 0.01 if not np.isnan(pivot_P[i]) else False
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = vol_spike[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG SETUP: Near support pivot + RSI divergence/oversold + HTF bull
        if near_S1 or near_S2 or near_P:
            # Need either divergence OR oversold RSI
            if (div_bull[i] or rsi_oversold) and htf_1d_bull:
                # Volume confirmation preferred but not required for pivot bounce
                if vol_confirm or rsi_oversold:
                    desired_signal = SIZE_STRONG if htf_1w_bull else SIZE_BASE
        
        # SHORT SETUP: Near resistance pivot + RSI divergence/overbought + HTF bear
        elif near_R1 or near_R2 or near_P:
            # Need either divergence OR overbought RSI
            if (div_bear[i] or rsi_overbought) and htf_1d_bear:
                # Volume confirmation preferred but not required for pivot rejection
                if vol_confirm or rsi_overbought:
                    desired_signal = -SIZE_STRONG if htf_1w_bear else -SIZE_BASE
        
        # BREAKOUT SETUP: Price breaks pivot with volume + HTF alignment
        if close[i] > pivot_P[i] and htf_1d_bull and hma_bull and vol_confirm:
            desired_signal = SIZE_STRONG if htf_1w_bull else SIZE_BASE
        
        if close[i] < pivot_P[i] and htf_1d_bear and hma_bear and vol_confirm:
            if desired_signal > 0:
                pass  # Don't flip on same bar
            else:
                desired_signal = -SIZE_STRONG if htf_1w_bear else -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                # Set stoploss
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
        
        signals[i] = final_signal
    
    return signals