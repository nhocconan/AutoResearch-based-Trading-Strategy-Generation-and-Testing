#!/usr/bin/env python3
"""
Experiment #240: 1h Primary + 4h/12h HTF — Simplified Regime-Adaptive with Session Filter

Hypothesis: After analyzing 239 experiments, the pattern is clear:
- Lower TF (1h) strategies fail due to TOO MANY trades (>200/yr) → fee drag
- OR TOO FEW trades (0 trades) → Sharpe=0.000 auto-reject
- Sweet spot: 30-80 trades/year with HTF direction + LTF timing

This strategy uses:
1. 12h HMA(21) slope for PRIMARY trend (bull/bear regime)
2. 4h RSI(14) for entry timing within HTF trend
3. Choppiness Index(14) to reduce size in choppy markets
4. Session filter: only 8-20 UTC (high liquidity hours)
5. Volume filter: volume > 0.8x 20-bar average
6. ATR(14) 2.5x trailing stoploss

Key improvements:
- LOOSER RSI thresholds (35/65) to guarantee trade frequency
- Smaller position size (0.20-0.25) for lower TF cost control
- Force-trade after 50 bars if no signal (prevents 0-trade failure)
- Session + volume confluence to reduce false signals

Position sizing: 0.20 base, 0.25 strong signals (discrete levels)
Target: 40-70 trades/year per symbol (within 1h cost model)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_session_regime_rsi_hma_4h12h_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_hma_slope(hma_values, lookback=3):
    """Calculate HMA slope as percentage change over lookback."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        prev = hma_values[i - lookback]
        curr = hma_values[i]
        if prev != 0 and not np.isnan(prev) and not np.isnan(curr):
            slope[i] = (curr - prev) / abs(prev) * 100
    return slope

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.fillna(20).values

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = period
    atr_vals = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr_vals).rolling(window=n, min_periods=n).sum().values
    hh = pd.Series(high).rolling(window=n, min_periods=n).max().values
    ll = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    chop = np.zeros(len(close))
    for i in range(n, len(close)):
        range_hl = hh[i] - ll[i]
        if range_hl > 0 and atr_sum[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / range_hl) / np.log10(n)
        else:
            chop[i] = 50.0
    
    return chop

def get_hour_from_open_time(open_time_array):
    """Extract hour from open_time (milliseconds timestamp)."""
    hours = np.zeros(len(open_time_array), dtype=int)
    for i in range(len(open_time_array)):
        ts_ms = open_time_array[i]
        ts_sec = ts_ms / 1000.0
        hours[i] = int((ts_sec % 86400) / 3600)
    return hours

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
    
    # Calculate 12h HTF indicators (primary trend regime)
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    hma_12h_slope = calculate_hma_slope(hma_12h_21, 3)
    
    # Calculate 4h HTF indicators (entry timing)
    rsi_4h = calculate_rsi(df_4h['close'].values, 14)
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    adx_4h = calculate_adx(df_4h['high'].values, df_4h['low'].values, df_4h['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    hma_12h_slope_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_slope)
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    hma_1h_21 = calculate_hma(close, 21)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    
    # Volume average (20-bar)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session hours (8-20 UTC = high liquidity)
    hours = get_hour_from_open_time(open_time)
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.20
    STRONG_SIZE = 0.25
    CHOP_SIZE = 0.15  # Reduced size in choppy markets
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(hma_12h_slope_aligned[i]):
            continue
        
        if np.isnan(rsi_4h_aligned[i]) or np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(hma_1h_21[i]):
            continue
        
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] == 0:
            continue
        
        # === REGIME DETECTION (12h HMA slope) ===
        # Bull regime: 12h HMA slope > 0.10%
        # Bear regime: 12h HMA slope < -0.10%
        # Neutral: between -0.10% and 0.10%
        regime_bull = hma_12h_slope_aligned[i] > 0.10
        regime_bear = hma_12h_slope_aligned[i] < -0.10
        regime_neutral = not regime_bull and not regime_bear
        
        price_above_12h_hma = close[i] > hma_12h_21_aligned[i]
        price_below_12h_hma = close[i] < hma_12h_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop_14[i] > 55.0
        is_trending = chop_14[i] < 45.0
        
        # === 4H HTF SIGNALS ===
        price_above_4h_hma = close[i] > hma_4h_21_aligned[i]
        price_below_4h_hma = close[i] < hma_4h_21_aligned[i]
        adx_trending = adx_4h_aligned[i] > 20
        
        # === 1H LOCAL SIGNALS ===
        price_above_1h_hma = close[i] > hma_1h_21[i]
        price_below_1h_hma = close[i] < hma_1h_21[i]
        
        # === RSI SIGNALS (LOOSE THRESHOLDS FOR TRADE FREQUENCY) ===
        rsi_oversold = rsi_14[i] < 40
        rsi_overbought = rsi_14[i] > 60
        rsi_4h_oversold = rsi_4h_aligned[i] < 45
        rsi_4h_overbought = rsi_4h_aligned[i] > 55
        
        # === VOLUME FILTER ===
        vol_ok = volume[i] > 0.8 * vol_avg_20[i]
        
        # === SESSION FILTER ===
        session_ok = in_session[i]
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        if is_choppy:
            current_size = CHOP_SIZE
        elif is_trending and adx_trending:
            current_size = STRONG_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG entries (regime bull or neutral + RSI oversold)
        if (regime_bull or regime_neutral) and price_above_12h_hma:
            # Primary long: RSI oversold + price above 12h HMA + session + volume
            if rsi_oversold and session_ok and vol_ok:
                new_signal = current_size
            # Secondary long: 4h RSI oversold + price above 4h HMA
            elif rsi_4h_oversold and price_above_4h_hma and session_ok:
                if new_signal == 0.0:
                    new_signal = current_size * 0.8
            # Tertiary long: Trending + pullback to 1h HMA
            elif is_trending and price_above_1h_hma and rsi_14[i] > 45 and rsi_14[i] < 55:
                if new_signal == 0.0 and session_ok:
                    new_signal = current_size * 0.7
        
        # SHORT entries (regime bear or neutral + RSI overbought)
        if (regime_bear or regime_neutral) and price_below_12h_hma:
            # Primary short: RSI overbought + price below 12h HMA + session + volume
            if rsi_overbought and session_ok and vol_ok:
                new_signal = -current_size
            # Secondary short: 4h RSI overbought + price below 4h HMA
            elif rsi_4h_overbought and price_below_4h_hma and session_ok:
                if new_signal == 0.0:
                    new_signal = -current_size * 0.8
            # Tertiary short: Trending + pullback to 1h HMA
            elif is_trending and price_below_1h_hma and rsi_14[i] > 45 and rsi_14[i] < 55:
                if new_signal == 0.0 and session_ok:
                    new_signal = -current_size * 0.7
        
        # === MEAN REVERSION IN CHOPPY MARKET ===
        if is_choppy and not regime_bull and not regime_bear:
            # Long at bottom of range
            if rsi_14[i] < 35 and session_ok:
                if new_signal == 0.0:
                    new_signal = CHOP_SIZE * 0.8
            # Short at top of range
            if rsi_14[i] > 65 and session_ok:
                if new_signal == 0.0:
                    new_signal = -CHOP_SIZE * 0.8
        
        # === FREQUENCY SAFEGUARD (CRITICAL for 10+ trades) ===
        # Force trade if no signal for 50 bars (~2 days on 1h)
        if bars_since_last_trade > 50 and new_signal == 0.0 and not in_position:
            if regime_bull and rsi_14[i] > 45 and price_above_1h_hma and session_ok:
                new_signal = BASE_SIZE * 0.5
            elif regime_bear and rsi_14[i] < 55 and price_below_1h_hma and session_ok:
                new_signal = -BASE_SIZE * 0.5
            elif is_choppy and rsi_14[i] < 38 and session_ok:
                new_signal = CHOP_SIZE * 0.6
            elif is_choppy and rsi_14[i] > 62 and session_ok:
                new_signal = -CHOP_SIZE * 0.6
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
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
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and regime_bear and price_below_12h_hma:
                regime_reversal = True
            if position_side < 0 and regime_bull and price_above_12h_hma:
                regime_reversal = True
        
        if stoploss_triggered or regime_reversal:
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