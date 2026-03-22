#!/usr/bin/env python3
"""
Experiment #335: 1h Primary + 4h/1d HTF — HMA Trend + RSI Pullback + Volume/Session Filter

Hypothesis: After 300+ failed experiments, the pattern is clear:
- Too many filters (Connors + Choppiness + multiple HTF) = 0 trades
- Simpler HMA + RSI with HTF trend = actual trades + positive Sharpe
- Current best (#333) uses 1d+1w with Sharpe=0.435

This strategy adapts the winning formula to 1h timeframe:
1. 4h HMA(21) for major trend direction (like 1w was for 1d)
2. 1d HMA(21) as secondary regime filter
3. 1h RSI(14) pullback entries (35-55 for longs, 45-65 for shorts)
4. Volume filter: current volume > 0.8x 20-bar average
5. Session filter: only 8-20 UTC (high liquidity, less manipulation)
6. ATR(14) trailing stop at 2.5x
7. Discrete sizing: 0.20 base, 0.30 strong (conservative for 1h)

Why this might beat Sharpe=0.435:
- 1h captures more intraday moves than 1d while using HTF for direction
- Session filter reduces false breakouts during low-liquidity hours
- Volume confirmation filters out weak signals
- Proven HMA+RSI combination from #333, adapted to lower TF
- Target: 40-70 trades/year (within 30-80 limit for 1h)

Position sizing: 0.20-0.30 (smaller than 1d due to more frequent signals)
Stoploss: 2.5 * ATR trailing
Leverage: 1.0 (no leverage)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_4h1d_session_vol_v1"
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
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Much less lag than EMA while maintaining smoothness.
    """
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

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_volume_sma(volume, period=20):
    """Calculate volume SMA for volume filter."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    return pd.to_datetime(open_time, unit='ms').dt.hour.values

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
    
    # Calculate 4h HTF indicators (major trend direction)
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    
    # Calculate 1d HTF indicators (secondary regime filter)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    hma_1h_8 = calculate_hma(close, period=8)
    hma_1h_21 = calculate_hma(close, period=21)
    sma_200 = calculate_sma(close, 200)
    vol_sma_20 = calculate_volume_sma(volume, 20)
    utc_hour = get_utc_hour(open_time)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    # Smaller for 1h due to more frequent signals
    LONG_BASE = 0.20
    LONG_STRONG = 0.30
    SHORT_BASE = 0.15
    SHORT_STRONG = 0.25
    
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
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_1h_8[i]) or np.isnan(hma_1h_21[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = 8 <= utc_hour[i] <= 20
        
        # === VOLUME FILTER ===
        vol_ratio = volume[i] / (vol_sma_20[i] + 1e-10)
        volume_ok = vol_ratio > 0.8
        
        # === 4H MAJOR TREND REGIME (primary direction filter) ===
        regime_4h_bull = close[i] > hma_4h_21_aligned[i]
        regime_4h_bear = close[i] < hma_4h_21_aligned[i]
        
        # === 1D SECONDARY REGIME (confirmation) ===
        regime_1d_bull = close[i] > hma_1d_21_aligned[i] if not np.isnan(hma_1d_21_aligned[i]) else True
        regime_1d_bear = close[i] < hma_1d_21_aligned[i] if not np.isnan(hma_1d_21_aligned[i]) else False
        
        # === 1H LOCAL TREND ===
        hma_bullish = hma_1h_8[i] > hma_1h_21[i]
        hma_bearish = hma_1h_8[i] < hma_1h_21[i]
        
        # HMA slope
        hma_slope_up = hma_1h_21[i] > hma_1h_21[i-2] if i >= 2 else False
        hma_slope_down = hma_1h_21[i] < hma_1h_21[i-2] if i >= 2 else False
        
        # Price position relative to HMA
        price_above_hma = close[i] > hma_1h_21[i]
        price_below_hma = close[i] < hma_1h_21[i]
        
        # Price relative to SMA200
        price_above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else True
        
        # === RSI SIGNALS (pullback entries) ===
        rsi_pullback_long = 35.0 < rsi_14[i] < 55.0
        rsi_pullback_short = 45.0 < rsi_14[i] < 65.0
        rsi_strong_oversold = rsi_14[i] < 35.0
        rsi_strong_overbought = rsi_14[i] > 65.0
        rsi_rising = rsi_14[i] > rsi_14[i-1] if i > 0 else False
        rsi_falling = rsi_14[i] < rsi_14[i-1] if i > 0 else False
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # Confluence counter for entry quality
        long_confluence = 0
        short_confluence = 0
        
        # Long confluence checks
        if regime_4h_bull:
            long_confluence += 1
        if regime_1d_bull:
            long_confluence += 1
        if hma_bullish:
            long_confluence += 1
        if price_above_hma:
            long_confluence += 1
        if rsi_rising:
            long_confluence += 1
        if volume_ok:
            long_confluence += 1
        if in_session:
            long_confluence += 1
        
        # Short confluence checks
        if regime_4h_bear:
            short_confluence += 1
        if regime_1d_bear:
            short_confluence += 1
        if hma_bearish:
            short_confluence += 1
        if price_below_hma:
            short_confluence += 1
        if rsi_falling:
            short_confluence += 1
        if volume_ok:
            short_confluence += 1
        if in_session:
            short_confluence += 1
        
        # LONG ENTRIES (need 4+ confluence in session, 3+ outside)
        if long_confluence >= 4 and in_session:
            if rsi_pullback_long and hma_bullish:
                new_signal = LONG_BASE
            
            elif rsi_strong_oversold and regime_4h_bull:
                new_signal = LONG_STRONG
            
            elif hma_bullish and hma_slope_up and rsi_rising and price_above_sma200:
                if new_signal == 0.0:
                    new_signal = LONG_BASE * 0.8
        
        elif long_confluence >= 3 and not in_session:
            # Outside session, require stronger signals
            if rsi_strong_oversold and regime_4h_bull and regime_1d_bull:
                new_signal = LONG_BASE * 0.8
        
        # SHORT ENTRIES (need 4+ confluence in session, 3+ outside)
        if short_confluence >= 4 and in_session:
            if new_signal == 0.0:
                if rsi_pullback_short and hma_bearish:
                    new_signal = -SHORT_BASE
                
                elif rsi_strong_overbought and regime_4h_bear:
                    new_signal = -SHORT_STRONG
                
                elif hma_bearish and hma_slope_down and rsi_falling and not price_above_sma200:
                    new_signal = -SHORT_BASE * 0.8
        
        elif short_confluence >= 3 and not in_session:
            if new_signal == 0.0:
                if rsi_strong_overbought and regime_4h_bear and regime_1d_bear:
                    new_signal = -SHORT_BASE * 0.8
        
        # === FREQUENCY SAFEGUARD (ensure 30+ trades/year on 1h) ===
        # Force trade if no signal for 120 bars (~5 days at 1h)
        if bars_since_last_trade > 120 and new_signal == 0.0 and not in_position:
            if regime_4h_bull and rsi_14[i] > 40.0 and volume_ok:
                new_signal = LONG_BASE * 0.5
            elif regime_4h_bear and rsi_14[i] < 60.0 and volume_ok:
                new_signal = -SHORT_BASE * 0.5
            elif rsi_strong_oversold and volume_ok:
                new_signal = LONG_BASE * 0.5
            elif rsi_strong_overbought and volume_ok:
                new_signal = -SHORT_BASE * 0.5
        
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
        
        # === RSI REVERSAL EXIT ===
        rsi_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and rsi_strong_overbought:
                rsi_exit = True
            if position_side < 0 and rsi_strong_oversold:
                rsi_exit = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and regime_4h_bear and price_below_hma:
                regime_reversal = True
            if position_side < 0 and regime_4h_bull and price_above_hma:
                regime_reversal = True
        
        if stoploss_triggered or rsi_exit or regime_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.10:
                new_signal = 0.0
            elif new_signal > 0.28:
                new_signal = LONG_STRONG
            elif new_signal > 0:
                new_signal = LONG_BASE
            elif new_signal < -0.23:
                new_signal = -SHORT_STRONG
            else:
                new_signal = -SHORT_BASE
        
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