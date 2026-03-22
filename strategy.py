#!/usr/bin/env python3
"""
Experiment #305: 1h Primary + 4h/1d HTF — Simplified Trend Pullback Strategy

Hypothesis: Previous 1h strategies failed (Sharpe=0.000) because entry conditions were TOO STRICT.
This version uses SIMPLER logic with fewer confluence filters to ensure trades actually happen.

Key learnings from failures (#295, #298, #300):
- Too many filters = 0 trades = auto-reject
- Need 30-60 trades/year on 1h (not 5-10)
- Session filter helps but don't over-constrain

Strategy design:
1. 4h HMA(21) = primary trend direction (long-only above, short-only below)
2. 1d HMA(21) = meta-regime filter (reduce size in counter-trend)
3. 1h RSI(14) pullback = entry timing (RSI 35-45 for long, 55-65 for short)
4. Volume > 0.7x avg = confirmation (looser than 1.0x to allow more trades)
5. Session 8-20 UTC = avoid Asian chop (but don't require all 3 filters)

Entry logic (LOOSENED for trade generation):
- Long: 4h HMA bullish + RSI 35-50 + (volume OR session)
- Short: 4h HMA bearish + RSI 50-65 + (volume OR session)

Position sizing: 0.25 base, 0.35 strong (when 1d trend aligns)
Stoploss: 2.5 * ATR trailing
Target: 40-80 trades/year on 1h

Why this should work:
- Fewer confluence requirements = more trades
- RSI pullback in trend = high probability setup
- 4h trend filter prevents counter-trend trades
- 1d filter adjusts sizing, doesn't block entries
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_pullback_4h1d_simp_v1"
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
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_n)
    
    return hma.values

def calculate_volume_sma(volume, period=20):
    """Calculate volume SMA for volume filter."""
    vol_s = pd.Series(volume)
    vol_sma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_sma

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    return pd.to_datetime(open_time, unit='ms').hour

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
    
    # Calculate HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    vol_sma_20 = calculate_volume_sma(volume, 20)
    
    # Extract UTC hours
    utc_hours = np.array([get_utc_hour(ot) for ot in open_time])
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.35
    MIN_SIZE = 0.15
    
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
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(vol_sma_20[i]):
            continue
        
        # === 4H TREND DIRECTION (primary filter) ===
        # Price above 4h HMA = bullish trend (prefer longs)
        # Price below 4h HMA = bearish trend (prefer shorts)
        trend_4h_bull = close[i] > hma_4h_aligned[i]
        trend_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === 1D META-REGIME (sizing filter) ===
        # 1d trend aligned = strong conviction (use STRONG_SIZE)
        # 1d trend against = weak conviction (use BASE_SIZE or MIN_SIZE)
        trend_1d_bull = close[i] > hma_1d_aligned[i]
        trend_1d_bear = close[i] < hma_1d_aligned[i]
        
        trend_aligned_long = trend_4h_bull and trend_1d_bull
        trend_aligned_short = trend_4h_bear and trend_1d_bear
        
        # === VOLUME CONFIRMATION ===
        vol_ratio = volume[i] / (vol_sma_20[i] + 1e-10)
        vol_confirmed = vol_ratio > 0.7  # Looser threshold for more trades
        
        # === SESSION FILTER (8-20 UTC) ===
        in_session = 8 <= utc_hours[i] <= 20
        
        # === RSI PULLBACK ZONES (LOOSENED for trade generation) ===
        # Long: RSI 35-50 (pullback in uptrend)
        # Short: RSI 50-65 (pullback in downtrend)
        rsi_long_zone = 35.0 <= rsi_14[i] <= 50.0
        rsi_short_zone = 50.0 <= rsi_14[i] <= 65.0
        
        # RSI turning (momentum confirmation)
        rsi_rising = rsi_14[i] > rsi_14[i-1] if i >= 1 else False
        rsi_falling = rsi_14[i] < rsi_14[i-1] if i >= 1 else False
        
        # === ENTRY LOGIC (SIMPLIFIED - fewer confluence requirements) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES (4h trend bull + RSI pullback + volume OR session)
        if trend_4h_bull and rsi_long_zone:
            # Need at least ONE of: volume confirmation OR in session
            if vol_confirmed or in_session:
                # Additional confirmation: RSI rising (momentum turning up)
                if rsi_rising:
                    if trend_aligned_long:
                        new_signal = STRONG_SIZE
                    else:
                        new_signal = BASE_SIZE
        
        # SHORT ENTRIES (4h trend bear + RSI pullback + volume OR session)
        if trend_4h_bear and rsi_short_zone:
            # Need at least ONE of: volume confirmation OR in session
            if vol_confirmed or in_session:
                # Additional confirmation: RSI falling (momentum turning down)
                if rsi_falling:
                    if trend_aligned_short:
                        new_signal = -STRONG_SIZE
                    else:
                        new_signal = -BASE_SIZE
        
        # === TRADE FREQUENCY SAFEGUARD ===
        # If no trade for 72 bars (~3 days on 1h), force entry on simpler conditions
        if bars_since_last_trade > 72 and new_signal == 0.0 and not in_position:
            if trend_4h_bull and rsi_14[i] < 45 and (vol_confirmed or in_session):
                new_signal = MIN_SIZE
            elif trend_4h_bear and rsi_14[i] > 55 and (vol_confirmed or in_session):
                new_signal = -MIN_SIZE
        
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
        
        # === TREND REVERSAL EXIT ===
        trend_exit = False
        if in_position and position_side != 0:
            # Long position: exit when 4h trend turns bearish
            if position_side > 0 and trend_4h_bear:
                trend_exit = True
            # Short position: exit when 4h trend turns bullish
            if position_side < 0 and trend_4h_bull:
                trend_exit = True
        
        # === RSI EXTREME EXIT (take profit) ===
        rsi_exit = False
        if in_position and position_side != 0:
            # Long position: exit when RSI overbought
            if position_side > 0 and rsi_14[i] > 70:
                rsi_exit = True
            # Short position: exit when RSI oversold
            if position_side < 0 and rsi_14[i] < 30:
                rsi_exit = True
        
        if stoploss_triggered or trend_exit or rsi_exit:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.12:
                new_signal = 0.0
            elif new_signal > 0.30:
                new_signal = STRONG_SIZE
            elif new_signal > 0:
                new_signal = BASE_SIZE
            elif new_signal < -0.30:
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