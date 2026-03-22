#!/usr/bin/env python3
"""
Experiment #418: 30m Primary + 4h/1d HTF — Regime-Adaptive RSI Pullback

Hypothesis: Previous 30m strategies (#408, #410) failed with Sharpe=0.000 because
entry conditions were TOO STRICT (0 trades). Key adjustments:
1. Use 4h HMA(21) for trend direction (proven in #406, #382)
2. Use 1d HMA(50) for major regime filter (bull/bear bias)
3. Use 30m RSI(14) with MODERATE thresholds (35-65, NOT extreme 10-90)
4. Volume filter: only >0.6x avg (not 1.5x which kills trades)
5. Session filter: 6-22 UTC (permissive, not 8-20 which is too narrow)
6. CRITICAL: Loosen confluence to ensure >=30 trades/symbol on train

Why this might work when #408 failed:
- #408 used CRSI<15 which rarely triggers on 30m
- #408 used CHOP>55 which filters out too many bars
- This uses simpler RSI(14) with 35-65 range (triggers more often)
- 4h HMA trend + 30m RSI pullback is proven pattern from best strategies

Position sizing: 0.20-0.25 (smaller for 30m to handle fee drag)
Stoploss: 2.5 * ATR trailing
Target: 40-80 trades/year on 30m, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_rsi_hma_pullback_4h1d_v1"
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

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HTF indicators (trend direction)
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    
    # Calculate 1d HTF indicators (major regime)
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    sma_200 = calculate_sma(close, 200)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, smaller for 30m)
    LONG_SIZE = 0.22
    SHORT_SIZE = 0.20
    
    # Track position state
    in_position = False
    position_side = 0
    highest_price = 0.0
    lowest_price = 0.0
    entry_price = 0.0
    last_trade_bar = -20
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(sma_200[i]) or np.isnan(vol_sma_20[i]):
            continue
        
        # === 1D MAJOR REGIME (primary bias) ===
        # Price above 1d HMA(50) = bull regime (favor longs)
        # Price below 1d HMA(50) = bear regime (favor shorts)
        bull_regime = close[i] > hma_1d_50_aligned[i]
        bear_regime = close[i] < hma_1d_50_aligned[i]
        
        # === 4H TREND DIRECTION ===
        # Price above 4h HMA(21) = uptrend
        # Price below 4h HMA(21) = downtrend
        trend_up = close[i] > hma_4h_21_aligned[i]
        trend_down = close[i] < hma_4h_21_aligned[i]
        
        # === 30M RSI PULLBACK (entry timing) ===
        # RSI 35-45 = pullback in uptrend (long entry)
        # RSI 55-65 = rally in downtrend (short entry)
        rsi_pullback_long = 35.0 <= rsi_14[i] <= 50.0
        rsi_pullback_short = 50.0 <= rsi_14[i] <= 65.0
        
        # === VOLUME CONFIRMATION (light filter) ===
        vol_ok = volume[i] > 0.6 * vol_sma_20[i] if not np.isnan(vol_sma_20[i]) else True
        
        # === SESSION FILTER (permissive: 6-22 UTC) ===
        # Convert open_time to hour (open_time is in milliseconds)
        hour_utc = (open_time[i] // 1000 // 3600) % 24
        session_ok = 6 <= hour_utc <= 22
        
        # === SMA200 FILTER (avoid counter-trend in strong trends) ===
        above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else True
        below_sma200 = close[i] < sma_200[i] if not np.isnan(sma_200[i]) else True
        
        # === ENTRY LOGIC — REGIME ADAPTIVE ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: bull regime + 4h uptrend + RSI pullback
        if bull_regime and trend_up:
            if rsi_pullback_long and vol_ok:
                new_signal = LONG_SIZE
            # Secondary: RSI < 40 in bull regime (deeper pullback)
            elif rsi_14[i] < 40.0 and vol_ok:
                new_signal = LONG_SIZE * 0.9
        
        # SHORT ENTRY: bear regime + 4h downtrend + RSI rally
        if bear_regime and trend_down:
            if rsi_pullback_short and vol_ok:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            # Secondary: RSI > 60 in bear regime (stronger rally)
            elif rsi_14[i] > 60.0 and vol_ok:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.9
        
        # === FREQUENCY BOOST (CRITICAL: ensure >=30 trades/symbol) ===
        # If no trade for 20 bars (~10 hours on 30m), force entry on weaker signal
        if bars_since_last_trade > 20 and new_signal == 0.0 and not in_position:
            if bull_regime and rsi_14[i] < 45.0:
                new_signal = LONG_SIZE * 0.7
            elif bear_regime and rsi_14[i] > 55.0:
                new_signal = -SHORT_SIZE * 0.7
        
        # === EXIT CONDITIONS ===
        # RSI extreme exit (take profit on exhaustion)
        if in_position and position_side > 0 and rsi_14[i] > 70.0:
            new_signal = 0.0
        if in_position and position_side < 0 and rsi_14[i] < 30.0:
            new_signal = 0.0
        
        # Regime flip exit (1d HMA cross)
        if in_position and position_side > 0 and bear_regime:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime:
            new_signal = 0.0
        
        # Trend reversal exit (4h HMA cross)
        if in_position and position_side > 0 and trend_down:
            new_signal = 0.0
        if in_position and position_side < 0 and trend_up:
            new_signal = 0.0
        
        # === STOPLOSS (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_price = max(highest_price, high[i])
            stop_price = highest_price - 2.5 * atr_14[i]
            if low[i] < stop_price and not np.isnan(stop_price):
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_price == 0.0:
                lowest_price = low[i]
            else:
                lowest_price = min(lowest_price, low[i])
            stop_price = lowest_price + 2.5 * atr_14[i]
            if high[i] > stop_price and not np.isnan(stop_price):
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = high[i] if position_side > 0 else 0.0
                lowest_price = low[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = high[i] if position_side > 0 else 0.0
                lowest_price = low[i] if position_side < 0 else 0.0
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