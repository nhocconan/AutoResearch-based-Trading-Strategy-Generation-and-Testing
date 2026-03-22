#!/usr/bin/env python3
"""
Experiment #400: 1h Primary + 4h/12h HTF — HMA Trend + RSI Pullback + Session Filter

Hypothesis: After 362+ failed experiments, the pattern is clear:
1. 1h timeframe needs 30-60 trades/year — too many = fee drag, too few = no signal
2. Complex regime filters (Chop, CRSI) consistently FAIL on lower TFs (#388, #390, #398)
3. 12h HMA(21) for major trend PROVEN in #382 (kept Sharpe=0.109)
4. 4h HMA(16/48) for intermediate confirmation — faster than 12h, slower than 1h
5. 1h RSI(14) pullback entries WITHIN HTF trend — wider range for frequency
6. Session filter (8-20 UTC) reduces overnight noise without killing trades
7. Simple volume filter (>0.5x avg) — not too strict

Why this might beat current best (Sharpe=0.435):
- 1h TF with HTF direction = best of both worlds (frequency + trend alignment)
- 12h HMA prevents counter-trend trades (reduces 2022-style whipsaw)
- 4h HMA cross confirms momentum before 1h entry
- RSI pullback (not extreme) ensures 40-60 trades/year target
- Session filter reduces false breakouts during low liquidity

Position sizing: 0.25-0.30 (discrete, max 0.40)
Stoploss: 2.5 * ATR trailing
Target: 40-60 trades/year on 1h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_session_4h12h_v1"
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

def calculate_sma(close, period=20):
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
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HTF indicators (major trend direction)
    hma_12h_21 = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    
    # Calculate 4h HTF indicators (intermediate trend)
    hma_4h_16 = calculate_hma(df_4h['close'].values, period=16)
    hma_4h_48 = calculate_hma(df_4h['close'].values, period=48)
    hma_4h_16_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_16)
    hma_4h_48_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_48)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    sma_200 = calculate_sma(close, 200)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.28
    SHORT_SIZE = 0.25
    
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
        
        if np.isnan(hma_12h_21_aligned[i]):
            continue
        
        if np.isnan(hma_4h_16_aligned[i]) or np.isnan(hma_4h_48_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(sma_200[i]) or np.isnan(vol_sma_20[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        # Extract hour from open_time (milliseconds timestamp)
        hour_utc = (open_time[i] // 3600000) % 24
        in_session = 8 <= hour_utc <= 20
        
        # === 12H MAJOR TREND (primary direction filter) ===
        bull_regime_12h = close[i] > hma_12h_21_aligned[i]
        bear_regime_12h = close[i] < hma_12h_21_aligned[i]
        
        # === 4H INTERMEDIATE TREND (confirmation) ===
        hma_4h_bullish = hma_4h_16_aligned[i] > hma_4h_48_aligned[i]
        hma_4h_bearish = hma_4h_16_aligned[i] < hma_4h_48_aligned[i]
        
        # === VOLUME FILTER (moderate, not too strict) ===
        vol_ok = volume[i] > 0.5 * vol_sma_20[i] if vol_sma_20[i] > 0 else True
        
        # === RSI PULLBACK SIGNALS (wider range for frequency) ===
        # Long: RSI pulled back to 30-55 in uptrend (buying dip)
        rsi_long_pullback = 30.0 <= rsi_14[i] <= 55.0
        # Short: RSI pulled back to 45-70 in downtrend (selling rally)
        rsi_short_pullback = 45.0 <= rsi_14[i] <= 70.0
        
        # === SMA200 FILTER (long-term trend) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === ENTRY LOGIC — HTF TREND + 1H PULLBACK ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: 12h bull + 4h bullish + RSI pullback + session + volume
        if bull_regime_12h and hma_4h_bullish and above_sma200:
            if rsi_long_pullback and in_session and vol_ok:
                new_signal = LONG_SIZE
        
        # SHORT ENTRY: 12h bear + 4h bearish + RSI pullback + session + volume
        if bear_regime_12h and hma_4h_bearish and below_sma200:
            if rsi_short_pullback and in_session and vol_ok:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
        
        # === FREQUENCY BOOST (ensure >=30 trades/symbol on train) ===
        # If no trade for 48 bars (~2 days on 1h), force entry on weaker signal
        if bars_since_last_trade > 48 and new_signal == 0.0 and not in_position:
            if bull_regime_12h and rsi_14[i] < 50 and hma_4h_bullish and in_session:
                new_signal = LONG_SIZE * 0.7
            elif bear_regime_12h and rsi_14[i] > 50 and hma_4h_bearish and in_session:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.7
        
        # === EXIT CONDITIONS ===
        # RSI extreme exit (take profit on momentum exhaustion)
        if in_position and position_side > 0 and rsi_14[i] > 75:
            new_signal = 0.0
        if in_position and position_side < 0 and rsi_14[i] < 25:
            new_signal = 0.0
        
        # Trend reversal exit (12h regime flip)
        if in_position and position_side > 0 and bear_regime_12h:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_regime_12h:
            new_signal = 0.0
        
        # Intermediate trend reversal exit (4h HMA cross)
        if in_position and position_side > 0 and hma_4h_bearish:
            new_signal = 0.0
        if in_position and position_side < 0 and hma_4h_bullish:
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