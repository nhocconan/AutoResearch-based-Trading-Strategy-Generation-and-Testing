#!/usr/bin/env python3
"""
Experiment #750: 1h Primary + 4h/12h HTF — Regime-Adaptive RSI Pullback

Hypothesis: After 500+ failed strategies, clear patterns emerge for lower TF success:
1. 1h/30m strategies fail due to TOO MANY trades (>200/yr) → fee drag kills profit
2. Solution: Use HTF (4h/12h) for SIGNAL DIRECTION, 1h only for ENTRY TIMING
3. Choppiness Index successfully filters trend vs range regimes (ETH Sharpe +0.923 in #727)
4. RSI pullback within HTF trend works better than pure breakout (proven in mtf_hma_rsi_zscore_v1)
5. Session filter (8-20 UTC) + volume filter reduces trades by 60% while keeping quality
6. Discrete sizing (0.25) with 2.5x ATR stop controls drawdown

Strategy design:
1. 12h HMA(21) for primary trend bias (proven reliable across all conditions)
2. 4h Choppiness Index(14) for regime detection (trend CHOP<38 vs range CHOP>62)
3. 1h RSI(7) for entry timing (pullback to 35-45 in uptrend, 55-65 in downtrend)
4. Session filter: only trade 8-20 UTC (high liquidity, lower slippage)
5. Volume filter: current volume > 0.8x 20-bar average
6. ATR(14) trailing stop 2.5x for risk management
7. Discrete signals: 0.0, ±0.25 (smaller size for 1h to reduce fee impact)

Key differences from failed 1h strategies (#740, #745, #748):
- Stricter entry: requires HTF trend + regime + RSI + session + volume (5 confluence)
- Session filter eliminates 60% of low-quality trades
- Volume filter eliminates choppy low-volume entries
- Smaller position size (0.25 vs 0.35) for lower TF fee management
- Target: 30-60 trades/year (not 200+)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1h (target 30-60 trades/year with strict filters)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_rsi_pullback_chop_hma_4h12h_session_vol_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average - smoother and more responsive than EMA."""
    if len(series) < period:
        return np.full(len(series), np.nan)
    
    wma1 = pd.Series(series).ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma2 = pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hma_raw = 2 * wma1 - wma2
    hma = pd.Series(hma_raw).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures whether market is trending or ranging.
    CHOP > 61.8 = ranging (mean reversion)
    CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_sma(close, period):
    """Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def extract_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    return (open_time // 3600000) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate primary (1h) indicators
    rsi_1h = calculate_rsi(close, period=7)  # Faster RSI for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    sma_50_1h = calculate_sma(close, period=50)
    sma_200_1h = calculate_sma(close, period=200)
    vol_sma_20 = calculate_sma(volume, period=20)
    
    # Calculate and align HTF indicators
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    chop_4h_raw = calculate_choppiness(
        df_4h['high'].values,
        df_4h['low'].values,
        df_4h['close'].values,
        period=14
    )
    chop_4h_aligned = align_htf_to_ltf(prices, df_4h, chop_4h_raw)
    
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Smaller for 1h to reduce fee impact
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):  # Need buffer for all indicators + HTF alignment
        # Skip if indicators not ready
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or atr_1h[i] <= 1e-10:
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_4h_aligned[i]):
            continue
        if np.isnan(chop_4h_aligned[i]) or np.isnan(sma_50_1h[i]):
            continue
        if np.isnan(vol_sma_20[i]) or vol_sma_20[i] <= 1e-10:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        hour = extract_hour(open_time[i])
        session_ok = 8 <= hour <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_sma_20[i]
        
        # === TREND BIAS (12h HTF HMA) ===
        trend_12h_bullish = close[i] > hma_12h_aligned[i]
        trend_12h_bearish = close[i] < hma_12h_aligned[i]
        
        # === 4h HMA CONFIRMATION ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === REGIME DETECTION (4h Choppiness) ===
        trending_regime = chop_4h_aligned[i] < 38.2
        ranging_regime = chop_4h_aligned[i] > 61.8
        
        # === RSI ENTRY SIGNALS (1h) ===
        # In uptrend: buy pullback to RSI 35-45
        rsi_pullback_long = 35 <= rsi_1h[i] <= 48
        # In downtrend: sell rally to RSI 52-65
        rsi_rally_short = 52 <= rsi_1h[i] <= 65
        
        # Extreme RSI for counter-trend in ranging market
        rsi_extreme_low = rsi_1h[i] < 30
        rsi_extreme_high = rsi_1h[i] > 70
        
        # === SMA FILTER ===
        above_sma50 = close[i] > sma_50_1h[i]
        above_sma200 = close[i] > sma_200_1h[i]
        below_sma50 = close[i] < sma_50_1h[i]
        below_sma200 = close[i] < sma_200_1h[i]
        
        desired_signal = 0.0
        
        # === TRENDING REGIME (CHOP < 38.2) ===
        if trending_regime and session_ok and volume_ok:
            # Long: 12h bullish + 4h bullish + RSI pullback + above SMA50
            if (trend_12h_bullish and trend_4h_bullish and 
                rsi_pullback_long and above_sma50):
                desired_signal = BASE_SIZE
            
            # Short: 12h bearish + 4h bearish + RSI rally + below SMA50
            if (trend_12h_bearish and trend_4h_bearish and 
                rsi_rally_short and below_sma50):
                desired_signal = -BASE_SIZE
        
        # === RANGING REGIME (CHOP > 61.8) ===
        elif ranging_regime and session_ok and volume_ok:
            # Mean reversion long: extreme RSI + 12h bullish bias
            if rsi_extreme_low and trend_12h_bullish and above_sma200:
                desired_signal = BASE_SIZE * 0.8  # Smaller size for mean reversion
            
            # Mean reversion short: extreme RSI + 12h bearish bias
            if rsi_extreme_high and trend_12h_bearish and below_sma200:
                desired_signal = -BASE_SIZE * 0.8
        
        # === NEUTRAL REGIME (38.2 <= CHOP <= 61.8) ===
        else:
            # Conservative: only trade with strong 12h trend confirmation
            if session_ok and volume_ok:
                if trend_12h_bullish and trend_4h_bullish and rsi_1h[i] < 45 and above_sma50:
                    desired_signal = BASE_SIZE * 0.8
                
                if trend_12h_bearish and trend_4h_bearish and rsi_1h[i] > 55 and below_sma50:
                    desired_signal = -BASE_SIZE * 0.8
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 12h HMA still bullish
                if trend_12h_bullish:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 12h HMA still bearish
                if trend_12h_bearish:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 12h trend reverses bearish
            if trend_12h_bearish:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 12h trend reverses bullish
            if trend_12h_bullish:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= BASE_SIZE:
                desired_signal = BASE_SIZE
            else:
                desired_signal = BASE_SIZE * 0.8
        elif desired_signal < 0:
            if desired_signal <= -BASE_SIZE:
                desired_signal = -BASE_SIZE
            else:
                desired_signal = -BASE_SIZE * 0.8
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals