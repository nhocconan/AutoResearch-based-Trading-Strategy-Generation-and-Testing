#!/usr/bin/env python3
"""
Experiment #295: 1h Primary + 4h/1d HTF — KAMA Adaptive Trend + Volume + Session + ADX

Hypothesis: After 267 failed experiments, try a NEW combination not yet tested:
1. KAMA (Kaufman Adaptive Moving Average) — adapts to market noise, less whipsaw than EMA/HMA
2. ADX(14) for trend strength — only trade when ADX > 20 (real momentum exists)
3. Volume spike confirmation — volume > 1.5x 20-bar avg (institutional participation)
4. Session filter — only 8-20 UTC (institutional flow hours, avoid Asian chop)
5. 4h KAMA for PRIMARY trend direction, 1d ADX for regime strength
6. Asymmetric RSI thresholds — Long: RSI<35, Short: RSI>65 (crypto drops faster than rallies)
7. Strict 3+ confluence requirement — prevents overtrading on 1h timeframe

Why this might work when others failed:
- KAMA specifically designed for noisy markets (crypto perfect use case)
- Volume filter eliminates false breakouts (major issue in crypto)
- Session filter captures real institutional flow (8-20 UTC = London/NY overlap)
- ADX ensures we only trade when trend has actual strength
- 4h HTF for direction + 1h for timing = proven pattern from #292 success

Position sizing: 0.20 base, 0.30 strong conviction (conservative for 1h TF)
Target: 40-70 trades/year per symbol (appropriate for 1h with strict filters)
Stoploss: 2.5 * ATR trailing (tighter than daily, appropriate for hourly)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_kama_adx_vol_session_4h1d_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise — moves fast in trends, slow in chop.
    ER (Efficiency Ratio) = |Net Change| / Sum of Absolute Changes
    SC (Smoothing Constant) = [ER * (fast_sc - slow_sc) + slow_sc]^2
    """
    n = len(close)
    kama = np.zeros(n)
    
    close_s = pd.Series(close)
    
    # Calculate Efficiency Ratio
    net_change = close_s.diff(er_period).abs()
    sum_changes = close_s.diff().abs().rolling(window=er_period, min_periods=er_period).sum()
    
    er = net_change / (sum_changes + 1e-10)
    er = er.fillna(0).values
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Calculate KAMA
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """
    Calculate Average Directional Index (ADX).
    ADX > 25 = strong trend, ADX < 20 = weak/no trend
    """
    n = period
    
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Smoothed values
    atr = tr.ewm(span=n, min_periods=n, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=n, min_periods=n, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=n, min_periods=n, adjust=False).mean() / atr)
    
    # DX and ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=n, min_periods=n, adjust=False).mean()
    
    return adx.values, plus_di.values, minus_di.values

def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_ma

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def extract_hour_from_open_time(prices):
    """Extract UTC hour from open_time column."""
    # open_time is in milliseconds since epoch
    hours = (prices['open_time'].values // (1000 * 60 * 60)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Extract session hours (Rule: only trade 8-20 UTC)
    hours = extract_hour_from_open_time(prices)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HTF indicators (primary trend direction)
    kama_4h_21 = calculate_kama(df_4h['close'].values, er_period=10, fast_period=2, slow_period=30)
    kama_4h_50 = calculate_kama(df_4h['close'].values, er_period=10, fast_period=2, slow_period=50)
    
    # Calculate 1d HTF indicators (regime strength)
    adx_1d, _, _ = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    kama_4h_21_aligned = align_htf_to_ltf(prices, df_4h, kama_4h_21)
    kama_4h_50_aligned = align_htf_to_ltf(prices, df_4h, kama_4h_50)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    kama_1h_21 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    kama_1h_50 = calculate_kama(close, er_period=10, fast_period=2, slow_period=50)
    adx_1h, plus_di_1h, minus_di_1h = calculate_adx(high, low, close, 14)
    vol_ma_20 = calculate_volume_ma(volume, 20)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40, conservative for 1h)
    BASE_SIZE = 0.20
    STRONG_SIZE = 0.30
    MIN_SIZE = 0.15
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -20
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(kama_4h_21_aligned[i]) or np.isnan(kama_4h_50_aligned[i]):
            continue
        
        if np.isnan(adx_1d_aligned[i]) or np.isnan(adx_1h[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(kama_1h_21[i]):
            continue
        
        if np.isnan(vol_ma_20[i]) or vol_ma_20[i] == 0:
            continue
        
        # === 4H TREND REGIME (primary direction filter) ===
        # Bull: 4h KAMA21 > 4h KAMA50 (uptrend)
        # Bear: 4h KAMA21 < 4h KAMA50 (downtrend)
        trend_4h_bull = kama_4h_21_aligned[i] > kama_4h_50_aligned[i]
        trend_4h_bear = kama_4h_21_aligned[i] < kama_4h_50_aligned[i]
        
        # Price position relative to 4h KAMA
        price_above_4h_kama = close[i] > kama_4h_21_aligned[i]
        price_below_4h_kama = close[i] < kama_4h_21_aligned[i]
        
        # === 1D REGIME STRENGTH ===
        # ADX > 25 = strong trend regime, ADX < 20 = weak/choppy
        regime_strong = adx_1d_aligned[i] > 25.0
        regime_weak = adx_1d_aligned[i] < 20.0
        
        # === 1H LOCAL SIGNALS ===
        # KAMA trend on 1h
        kama_1h_bullish = kama_1h_21[i] > kama_1h_50[i]
        kama_1h_bearish = kama_1h_21[i] < kama_1h_50[i]
        
        # ADX momentum on 1h (only trade when ADX > 20)
        adx_momentum = adx_1h[i] > 20.0
        
        # DI crossover
        di_bullish = plus_di_1h[i] > minus_di_1h[i]
        di_bearish = plus_di_1h[i] < minus_di_1h[i]
        
        # === VOLUME CONFIRMATION ===
        # Volume spike = current volume > 1.5x 20-bar average
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        vol_normal = volume[i] > 0.8 * vol_ma_20[i]
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = (hours[i] >= 8) and (hours[i] <= 20)
        
        # === RSI THRESHOLDS (asymmetric for crypto) ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        rsi_extreme_oversold = rsi_14[i] < 25.0
        rsi_extreme_overbought = rsi_14[i] > 75.0
        rsi_neutral = (rsi_14[i] >= 40.0) and (rsi_14[i] <= 60.0)
        
        # === BOLLINGER BAND SIGNALS ===
        bb_break_lower = close[i] < bb_lower[i] * 1.002
        bb_break_upper = close[i] > bb_upper[i] * 0.998
        bb_revert_from_lower = (close[i] > bb_lower[i]) and (close[i-1] <= bb_lower[i-1]) if i > 0 else False
        bb_revert_from_upper = (close[i] < bb_upper[i]) and (close[i-1] >= bb_upper[i-1]) if i > 0 else False
        
        # === ENTRY LOGIC (3+ CONFLUENCE REQUIRED) ===
        new_signal = 0.0
        
        # LONG ENTRIES (require: 4h trend bull + 1h momentum + volume/session)
        if trend_4h_bull and price_above_4h_kama:
            confluence_count = 0
            
            # Confluence 1: 1h KAMA bullish
            if kama_1h_bullish:
                confluence_count += 1
            
            # Confluence 2: ADX momentum or DI bullish
            if adx_momentum and di_bullish:
                confluence_count += 1
            
            # Confluence 3: Volume confirmation
            if vol_spike or vol_normal:
                confluence_count += 1
            
            # Confluence 4: RSI confirmation (not overbought)
            if rsi_14[i] < 60.0:
                confluence_count += 1
            
            # Confluence 5: In session hours
            if in_session:
                confluence_count += 1
            
            # Strong long: 4+ confluence + volume spike
            if confluence_count >= 4 and vol_spike:
                new_signal = STRONG_SIZE
            # Normal long: 3+ confluence
            elif confluence_count >= 3:
                new_signal = BASE_SIZE
            # Mean revert long in weak regime: RSI extreme oversold
            elif regime_weak and rsi_extreme_oversold and bb_break_lower:
                new_signal = BASE_SIZE
        
        # SHORT ENTRIES (require: 4h trend bear + 1h momentum + volume/session)
        if trend_4h_bear and price_below_4h_kama:
            confluence_count = 0
            
            # Confluence 1: 1h KAMA bearish
            if kama_1h_bearish:
                confluence_count += 1
            
            # Confluence 2: ADX momentum or DI bearish
            if adx_momentum and di_bearish:
                confluence_count += 1
            
            # Confluence 3: Volume confirmation
            if vol_spike or vol_normal:
                confluence_count += 1
            
            # Confluence 4: RSI confirmation (not oversold)
            if rsi_14[i] > 40.0:
                confluence_count += 1
            
            # Confluence 5: In session hours
            if in_session:
                confluence_count += 1
            
            # Strong short: 4+ confluence + volume spike
            if confluence_count >= 4 and vol_spike:
                if new_signal == 0.0:
                    new_signal = -STRONG_SIZE
            # Normal short: 3+ confluence
            elif confluence_count >= 3:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
            # Mean revert short in weak regime: RSI extreme overbought
            elif regime_weak and rsi_extreme_overbought and bb_break_upper:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
        
        # === TRADE FREQUENCY SAFEGUARD (ensure 40+ trades/year on 1h) ===
        # Force trade if no signal for 60 bars (~60 hours = 2.5 days on 1h)
        bars_since_last_trade = i - last_trade_bar
        if bars_since_last_trade > 60 and new_signal == 0.0 and not in_position:
            # Only force if we have some directional bias
            if trend_4h_bull and kama_1h_bullish and rsi_14[i] < 55 and in_session:
                new_signal = MIN_SIZE
            elif trend_4h_bear and kama_1h_bearish and rsi_14[i] > 45 and in_session:
                new_signal = -MIN_SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing (appropriate for 1h) ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Long position but 4h trend turns bearish
            if position_side > 0 and trend_4h_bear and kama_1h_bearish:
                trend_reversal = True
            # Short position but 4h trend turns bullish
            if position_side < 0 and trend_4h_bull and kama_1h_bullish:
                trend_reversal = True
        
        # === RSI EXTREME EXIT (take profit) ===
        rsi_exit = False
        if in_position and position_side != 0:
            # Long position: exit when RSI extremely overbought
            if position_side > 0 and rsi_14[i] > 75.0:
                rsi_exit = True
            # Short position: exit when RSI extremely oversold
            if position_side < 0 and rsi_14[i] < 25.0:
                rsi_exit = True
        
        if stoploss_triggered or trend_reversal or rsi_exit:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn - Rule 4) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.15:
                new_signal = 0.0
            elif new_signal > 0.25:
                new_signal = STRONG_SIZE
            elif new_signal > 0:
                new_signal = BASE_SIZE
            elif new_signal < -0.25:
                new_signal = -STRONG_SIZE
            else:
                new_signal = -BASE_SIZE
        
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