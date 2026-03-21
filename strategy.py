#!/usr/bin/env python3
"""
Experiment #026: 30m MACD Momentum + 4h HMA Trend + RSI Entry Timing
Hypothesis: 30m timeframe captures intraday swings while 4h HMA provides major trend filter.
MACD histogram gives momentum direction before price moves. RSI extremes provide entry timing.
Volume surge confirms breakout validity. This hybrid approach should beat pure trend following.
Position sizing 0.30 with 2.5x ATR stoploss. Multiple entry triggers ensure ≥10 trades.
Relaxed RSI thresholds (35-65) to avoid 0-trade failure mode seen in experiments #016-017.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_macd_4h_hma_rsi_v1"
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
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return wma3.values

def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD indicator with histogram."""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line.values, signal_line.values, histogram.values

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_sma(values, period):
    """Calculate Simple Moving Average."""
    return pd.Series(values).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Also load 1d for major regime filter
    df_1d = get_htf_data(prices, '1d')
    hma_1d = calculate_hma(df_1d['close'].values, 50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
    
    # Additional trend indicators
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    # Volume SMA for confirmation
    vol_sma = calculate_sma(volume, 20)
    vol_sma = np.nan_to_num(vol_sma, nan=np.nanmean(volume))
    
    signals = np.zeros(n)
    SIZE = 0.30
    HALF_SIZE = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(200, n):
        # 4h trend filter (major regime)
        hma_4h_valid = not np.isnan(hma_4h_aligned[i]) and hma_4h_aligned[i] > 0
        trend_4h_bullish = hma_4h_valid and close[i] > hma_4h_aligned[i]
        trend_4h_bearish = hma_4h_valid and close[i] < hma_4h_aligned[i]
        
        # 1d trend filter (major regime)
        hma_1d_valid = not np.isnan(hma_1d_aligned[i]) and hma_1d_aligned[i] > 0
        trend_1d_bullish = hma_1d_valid and close[i] > hma_1d_aligned[i]
        trend_1d_bearish = hma_1d_valid and close[i] < hma_1d_aligned[i]
        
        # MACD momentum signals
        macd_bullish = macd_hist[i] > 0 and macd_hist[i] > macd_hist[i-1]
        macd_bearish = macd_hist[i] < 0 and macd_hist[i] < macd_hist[i-1]
        macd_cross_long = macd_hist[i] > 0 and macd_hist[i-1] <= 0
        macd_cross_short = macd_hist[i] < 0 and macd_hist[i-1] >= 0
        
        # RSI entry timing (relaxed thresholds for more trades)
        rsi_bullish = rsi[i] > 40 and rsi[i] < 70
        rsi_bearish = rsi[i] > 30 and rsi[i] < 60
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_rising = rsi[i] > rsi[i-2] if i > 2 else True
        rsi_falling = rsi[i] < rsi[i-2] if i > 2 else True
        
        # HMA trend on 30m
        hma_trend_long = hma_21[i] > hma_50[i] and hma_21[i] > sma_200[i]
        hma_trend_short = hma_21[i] < hma_50[i] and hma_21[i] < sma_200[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_sma[i] * 1.2 if vol_sma[i] > 0 else True
        vol_neutral = volume[i] > vol_sma[i] * 0.8 if vol_sma[i] > 0 else True
        
        # Price position
        price_above_hma21 = close[i] > hma_21[i]
        price_below_hma21 = close[i] < hma_21[i]
        price_above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else True
        price_below_sma200 = close[i] < sma_200[i] if not np.isnan(sma_200[i]) else True
        
        # Entry logic - MULTIPLE triggers to ensure trades (Rule 9)
        new_signal = 0.0
        
        # LONG ENTRY TRIGGERS (any one can trigger)
        # Trigger 1: MACD cross long + 4h bullish + RSI ok
        if macd_cross_long and trend_4h_bullish and rsi_bullish:
            new_signal = SIZE
        # Trigger 2: MACD bullish + HMA trend + volume + 4h support
        elif macd_bullish and hma_trend_long and vol_neutral and trend_4h_bullish:
            new_signal = SIZE
        # Trigger 3: RSI oversold bounce + 4h bullish + MACD improving
        elif rsi_oversold and rsi_rising and trend_4h_bullish and macd_hist[i] > macd_hist[i-2]:
            new_signal = SIZE
        # Trigger 4: Price above SMA200 + MACD bullish + 4h trend
        elif price_above_sma200 and macd_bullish and trend_4h_bullish and price_above_hma21:
            new_signal = SIZE
        # Trigger 5: 1d bullish regime + MACD cross (strong trend continuation)
        elif trend_1d_bullish and macd_cross_long and vol_confirm:
            new_signal = SIZE
        
        # SHORT ENTRY TRIGGERS (any one can trigger)
        # Trigger 1: MACD cross short + 4h bearish + RSI ok
        if macd_cross_short and trend_4h_bearish and rsi_bearish:
            new_signal = -SIZE
        # Trigger 2: MACD bearish + HMA trend + volume + 4h resistance
        elif macd_bearish and hma_trend_short and vol_neutral and trend_4h_bearish:
            new_signal = -SIZE
        # Trigger 3: RSI overbought drop + 4h bearish + MACD worsening
        elif rsi_overbought and rsi_falling and trend_4h_bearish and macd_hist[i] < macd_hist[i-2]:
            new_signal = -SIZE
        # Trigger 4: Price below SMA200 + MACD bearish + 4h trend
        elif price_below_sma200 and macd_bearish and trend_4h_bearish and price_below_hma21:
            new_signal = -SIZE
        # Trigger 5: 1d bearish regime + MACD cross (strong trend continuation)
        elif trend_1d_bearish and macd_cross_short and vol_confirm:
            new_signal = -SIZE
        
        # Stoploss logic (Rule 6) - ATR based with trailing
        if position_side > 0 and entry_price > 0:
            # Update highest since entry for trailing
            if close[i] > highest_since_entry:
                highest_since_entry = close[i]
            
            stop_loss = entry_price - 2.5 * atr[i]
            trail_stop = highest_since_entry - 2.5 * atr[i]
            effective_stop = max(stop_loss, trail_stop)
            
            if close[i] < effective_stop:
                new_signal = 0.0  # Stoploss hit
            # Take partial profit at 3R
            elif close[i] > entry_price + 3.0 * atr[i] and signals[i-1] == SIZE:
                new_signal = HALF_SIZE
        
        if position_side < 0 and entry_price > 0:
            # Update lowest since entry for trailing
            if lowest_since_entry == 0 or close[i] < lowest_since_entry:
                lowest_since_entry = close[i]
            
            stop_loss = entry_price + 2.5 * atr[i]
            trail_stop = lowest_since_entry + 2.5 * atr[i]
            effective_stop = min(stop_loss, trail_stop)
            
            if close[i] > effective_stop:
                new_signal = 0.0  # Stoploss hit
            # Take partial profit at 3R
            elif close[i] < entry_price - 3.0 * atr[i] and signals[i-1] == -SIZE:
                new_signal = -HALF_SIZE
        
        # Update position tracking
        if new_signal != 0 and position_side == 0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            highest_since_entry = close[i] if position_side > 0 else 0.0
            lowest_since_entry = close[i] if position_side < 0 else 0.0
        elif new_signal != 0 and position_side != 0:
            if np.sign(new_signal) != position_side:
                entry_price = close[i]
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        elif new_signal == 0 and position_side != 0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals