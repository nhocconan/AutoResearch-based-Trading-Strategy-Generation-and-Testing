#!/usr/bin/env python3
"""
Experiment #368: 30m Primary + 4h/1d HTF — Strict Confluence Trend-Follow

Hypothesis: After 367 experiments, the clearest pattern is:
1. Lower TF (30m) FAILS with too many trades → fee drag destroys edge
2. Complex regime detection adds no value (exp #356, #365 both negative Sharpe)
3. SIMPLICITY wins: current best is simple 1d HMA + 1w RSI (Sharpe=0.435)
4. For 30m to work: MUST use HTF for DIRECTION, 30m only for ENTRY TIMING
5. LONG-ONLY bias matches crypto's structural upward drift (avoid short whipsaw)
6. Require 4+ confluence filters to limit trades to 30-60/year target

Strategy Design:
- 4h HMA(21) slope > 0 for trend direction (primary filter)
- 1d price > HMA(21) for major bull regime (secondary filter)
- 30m RSI(14) pullback to 35-45 zone for entry timing (precision)
- Volume > 1.3x 20-bar avg for confirmation (avoids fake breakouts)
- Session filter 8-20 UTC only (highest liquidity, avoids Asian chop)
- Choppiness Index < 55 to avoid range markets (trend-only)

Position Sizing:
- LONG_BASE = 0.20 (conservative for lower TF)
- LONG_STRONG = 0.25 (when all 5 filters align)
- NO SHORTS (crypto long bias, shorts whipsaw in 2021-2024)
- Stoploss: 2.5 * ATR(14) trailing

Expected Trade Frequency: 30-60/year on 30m (strict filters)
Target Sharpe: > 0.5 (beat current best 0.435)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_longonly_confluence_4h1d_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/ranging market
    CHOP < 38.2 = trending market
    """
    atr_vals = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr_vals).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    chop[np.isnan(chop)] = 50.0
    
    return chop

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / (vol_avg + 1e-10)
    return vol_ratio

def get_hour_from_opentime(prices):
    """Extract hour from open_time for session filter."""
    # open_time is in milliseconds since epoch
    timestamps = pd.to_datetime(prices["open_time"].values, unit='ms', utc=True)
    hours = timestamps.hour.values
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    
    # Calculate 4h HMA slope (direction)
    hma_4h_slope = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(hma_4h_21_aligned[i]) and not np.isnan(hma_4h_21_aligned[i-1]):
            hma_4h_slope[i] = hma_4h_21_aligned[i] - hma_4h_21_aligned[i-1]
        else:
            hma_4h_slope[i] = 0.0
    
    # Calculate 1d HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    sma_200 = calculate_sma(close, 200)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    # Session filter (8-20 UTC)
    hours = get_hour_from_opentime(prices)
    session_valid = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, conservative for 30m)
    LONG_BASE = 0.20
    LONG_STRONG = 0.25
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            continue
        
        # === FILTER 1: 4H TREND DIRECTION (HMA slope > 0) ===
        trend_4h_bull = hma_4h_slope[i] > 0
        
        # === FILTER 2: 1D MAJOR REGIME (price > HMA21) ===
        regime_1d_bull = close[i] > hma_1d_21_aligned[i]
        
        # === FILTER 3: 30M RSI PULLBACK (35-45 zone for longs) ===
        rsi_pullback = 35.0 <= rsi_14[i] <= 48.0
        
        # === FILTER 4: VOLUME CONFIRMATION (> 1.3x avg) ===
        volume_confirmed = vol_ratio[i] > 1.2
        
        # === FILTER 5: CHOPPINESS < 55 (avoid range markets) ===
        trending_market = chop_14[i] < 55.0
        
        # === FILTER 6: SESSION (8-20 UTC only) ===
        session_ok = session_valid[i]
        
        # === FILTER 7: PRICE ABOVE SMA200 (long-term bull) ===
        above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else True
        
        # === ENTRY LOGIC: LONG ONLY (crypto bias) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # Count confluence factors
        confluence_count = sum([
            trend_4h_bull,
            regime_1d_bull,
            rsi_pullback,
            volume_confirmed,
            trending_market,
            session_ok,
            above_sma200
        ])
        
        # STRICT ENTRY: require 5+ of 7 filters for base position
        # require 6+ of 7 filters for strong position
        if confluence_count >= 6:
            new_signal = LONG_STRONG
        elif confluence_count >= 5:
            new_signal = LONG_BASE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trade for 30 bars (~15 hours on 30m), allow relaxed entry
        if bars_since_last_trade > 30 and new_signal == 0.0 and not in_position:
            if trend_4h_bull and regime_1d_bull and rsi_14[i] < 45.0:
                new_signal = LONG_BASE * 0.7
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if close[i] > highest_price:
                highest_price = close[i]
            stoploss_price = highest_price - 2.5 * atr_14[i]
            if close[i] < stoploss_price:
                stoploss_triggered = True
        
        # === RSI OVERBOUGHT EXIT ===
        rsi_exit = False
        if in_position and position_side > 0:
            if rsi_14[i] > 72.0:
                rsi_exit = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side > 0:
            if hma_4h_slope[i] < 0:
                trend_reversal = True
            if close[i] < hma_4h_21_aligned[i]:
                trend_reversal = True
        
        if stoploss_triggered or rsi_exit or trend_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if new_signal > 0.23:
                new_signal = LONG_STRONG
            else:
                new_signal = LONG_BASE
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = 1  # long only
                entry_price = close[i]
                highest_price = close[i]
                last_trade_bar = i
            elif new_signal == 0.0:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals