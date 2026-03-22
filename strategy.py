#!/usr/bin/env python3
"""
Experiment #420: 1h Primary + 4h/12h HTF — HMA Trend + RSI Pullback + Volume Session

Hypothesis: After 381 failed experiments, the pattern is clear:
1. 1h strategies fail due to EITHER too many trades (>200/yr) OR too few (0 trades)
2. Key insight: Use 4h/12h for TREND DIRECTION only, 1h for ENTRY TIMING
3. This gives HTF trade frequency (30-60/yr) with 1h execution precision
4. Previous 1h failures (#410, #415, #418) had entry conditions TOO STRICT

Why this might beat current best (Sharpe=0.435):
- 4h HMA(21) for major trend bias (proven in #382, #405)
- 1h RSI(14) pullback entries within HTF trend (not counter-trend)
- Volume confirmation (>0.8x avg) filters false breakouts
- Session filter (8-20 UTC) avoids overnight noise/whipsaw
- 12h Choppiness for regime detection (range vs trend)
- ATR 2.5x trailing stop protects in crash scenarios

Position sizing: 0.25 (smaller for 1h TF to reduce fee impact)
Target: 40-60 trades/year on 1h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_vol_session_4h12h_v1"
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
    Calculate Choppiness Index.
    CHOP > 61.8 = ranging market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    hh = high_s.rolling(window=period, min_periods=period).max()
    ll = low_s.rolling(window=period, min_periods=period).min()
    
    atr = calculate_atr(high, low, close, period)
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum()
    
    chop = 100.0 * np.log10((hh - ll).values / (atr_sum.values + 1e-10)) / np.log10(period)
    
    return chop

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean()
    vol_ratio = vol_s / (vol_avg + 1e-10)
    return vol_ratio.values

def get_hour_from_open_time(open_time):
    """Extract hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    hours = (open_time // (1000 * 60 * 60)) % 24
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
    
    # Calculate 4h HTF indicators (major trend direction)
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_48 = calculate_hma(df_4h['close'].values, period=48)
    
    # Calculate 12h HTF indicators (regime detection)
    chop_12h = calculate_choppiness(
        df_12h['high'].values,
        df_12h['low'].values,
        df_12h['close'].values,
        period=14
    )
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_48_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_48)
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    hma_1h_16 = calculate_hma(close, period=16)
    hma_1h_48 = calculate_hma(close, period=48)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    # Extract hour for session filter
    hours = np.array([get_hour_from_open_time(ot) for ot in open_time])
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40, smaller for 1h)
    SIZE = 0.25
    
    # Track position state
    in_position = False
    position_side = 0
    highest_price = 0.0
    lowest_price = 0.0
    entry_price = 0.0
    last_trade_bar = -50
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_48_aligned[i]):
            continue
        
        if np.isnan(chop_12h_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_1h_16[i]) or np.isnan(hma_1h_48[i]):
            continue
        
        if np.isnan(vol_ratio[i]):
            continue
        
        # === 4H MAJOR TREND (primary direction filter) ===
        # Price above 4h HMA(21) = bull bias (favor longs)
        # Price below 4h HMA(21) = bear bias (favor shorts)
        bull_regime = close[i] > hma_4h_21_aligned[i]
        bear_regime = close[i] < hma_4h_21_aligned[i]
        
        # 4h HMA crossover confirmation
        hma_4h_bullish = hma_4h_21_aligned[i] > hma_4h_48_aligned[i]
        hma_4h_bearish = hma_4h_21_aligned[i] < hma_4h_48_aligned[i]
        
        # === 12H CHOPPINESS REGIME ===
        # CHOP > 55 = ranging (prefer mean reversion entries)
        # CHOP < 45 = trending (prefer breakout entries)
        is_choppy = chop_12h_aligned[i] > 55.0
        is_trending = chop_12h_aligned[i] < 45.0
        
        # === 1H LOCAL TREND (entry timing) ===
        hma_1h_bullish = hma_1h_16[i] > hma_1h_48[i]
        hma_1h_bearish = hma_1h_16[i] < hma_1h_48[i]
        
        # === RSI PULLBACK SIGNALS ===
        # In bull regime: look for RSI pullback to 35-45 zone
        # In bear regime: look for RSI rally to 55-65 zone
        rsi_pullback_long = rsi_14[i] < 45.0 and rsi_14[i] > 25.0
        rsi_pullback_short = rsi_14[i] > 55.0 and rsi_14[i] < 75.0
        
        # RSI extreme (mean reversion in choppy market)
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = vol_ratio[i] > 0.8
        
        # === SESSION FILTER (8-20 UTC) ===
        # Only trade during active hours to avoid overnight noise
        in_session = (hours[i] >= 8) and (hours[i] <= 20)
        
        # === ENTRY LOGIC — REGIME ADAPTIVE ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY conditions (must have 3+ confluence)
        if bull_regime:
            # Confluence 1: 4h trend bullish
            # Confluence 2: 1h HMA bullish OR RSI pullback
            # Confluence 3: Volume confirmed
            # Confluence 4: In session
            
            long_score = 0
            if hma_4h_bullish:
                long_score += 1
            if hma_1h_bullish or rsi_pullback_long:
                long_score += 1
            if vol_confirmed:
                long_score += 1
            if in_session:
                long_score += 1
            
            # Need 3+ confluence for long entry
            if long_score >= 3:
                new_signal = SIZE
        
        # SHORT ENTRY conditions (must have 3+ confluence)
        if bear_regime:
            short_score = 0
            if hma_4h_bearish:
                short_score += 1
            if hma_1h_bearish or rsi_pullback_short:
                short_score += 1
            if vol_confirmed:
                short_score += 1
            if in_session:
                short_score += 1
            
            # Need 3+ confluence for short entry
            if short_score >= 3:
                new_signal = -SIZE
        
        # === FREQUENCY BOOST (ensure >=30 trades/symbol on train) ===
        # If no trade for 30 bars (~30 hours on 1h), relax entry conditions
        if bars_since_last_trade > 30 and new_signal == 0.0 and not in_position:
            if bull_regime and rsi_14[i] < 50.0 and vol_confirmed:
                new_signal = SIZE * 0.8
            elif bear_regime and rsi_14[i] > 50.0 and vol_confirmed:
                new_signal = -SIZE * 0.8
        
        # === EXIT CONDITIONS ===
        # RSI extreme exit (take profit)
        if in_position and position_side > 0 and rsi_14[i] > 70.0:
            new_signal = 0.0
        if in_position and position_side < 0 and rsi_14[i] < 30.0:
            new_signal = 0.0
        
        # Trend reversal exit (4h regime flip)
        if in_position and position_side > 0 and bear_regime:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime:
            new_signal = 0.0
        
        # Local trend reversal exit (1h HMA cross)
        if in_position and position_side > 0 and hma_1h_bearish and not rsi_pullback_long:
            new_signal = 0.0
        if in_position and position_side < 0 and hma_1h_bullish and not rsi_pullback_short:
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