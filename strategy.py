#!/usr/bin/env python3
"""
Experiment #211: 15m MACD Momentum + 4h HMA Trend + 1h RSI Pullback + Volume Filter

Hypothesis: 15m timeframe captures intraday momentum moves while 4h HMA provides
stable higher-timeframe bias. MACD histogram on 15m identifies momentum shifts,
1h RSI pullback ensures we enter on retracements (not chasing), and volume
spike confirms genuine breakouts. This should work better than pure trend or
pure mean-reversion approaches that failed in recent experiments.

Why 15m might work:
- 15m bars = 96 per day, captures intraday swings without 5m noise
- MACD histogram shows momentum acceleration/deceleration
- 4h HMA filter prevents counter-trend trades in strong trends
- 1h RSI pullback (RSI 40-60 in trend direction) = better entry timing
- Volume spike (>1.5x avg) confirms genuine moves, filters false breakouts
- Conservative sizing (0.25) controls drawdown in volatile periods

Learning from failures:
- #199 (15m KAMA): Sharpe=-4.763 - pure trend whipsaws on 15m
- #205 (15m EMA pullback): Sharpe=-3.678 - pullback alone insufficient
- #207 (1h RSI mean rev): Sharpe=-9.084 - mean reversion fails in trends
- Need MULTI-SIGNAL confirmation: HTF trend + LTF momentum + volume + pullback
- 15m needs STRONGER HTF filter than 4h/12h strategies

Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop), 1h for RSI pullback
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_macd_4h_hma_1h_rsi_vol_atr_v1"
timeframe = "15m"
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

def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram."""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line.values, signal_line.values, histogram.values

def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def calculate_volume_spike(volume, period=20, threshold=1.5):
    """Calculate if volume is spiking above average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.ewm(span=period, min_periods=period, adjust=False).mean()
    vol_ratio = volume / (vol_avg.values + 1e-10)
    return vol_ratio > threshold

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate 4h HMA for trend bias
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h RSI for pullback detection
    rsi_1h = calculate_rsi(df_1h['close'].values, 14)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
    rsi_15m = calculate_rsi(close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    vol_spike = calculate_volume_spike(volume, 20, 1.5)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(rsi_1h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(macd_hist[i]) or np.isnan(rsi_15m[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = higher timeframe trend bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === 1H RSI PULLBACK DETECTION ===
        # In uptrend: look for RSI pullback to 40-55 zone
        # In downtrend: look for RSI pullback to 45-60 zone
        rsi_pullback_long = 40 <= rsi_1h_aligned[i] <= 55
        rsi_pullback_short = 45 <= rsi_1h_aligned[i] <= 60
        
        # === 15M MACD MOMENTUM ===
        # MACD histogram crossing above 0 = bullish momentum
        # MACD histogram crossing below 0 = bearish momentum
        macd_bullish = macd_hist[i] > 0 and macd_hist[i] > macd_hist[i-1]
        macd_bearish = macd_hist[i] < 0 and macd_hist[i] < macd_hist[i-1]
        
        # === EMA STRUCTURE ===
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # === VOLUME CONFIRMATION ===
        # Volume spike confirms genuine moves
        vol_confirmed = vol_spike[i]
        
        # === 15M RSI MOMENTUM ===
        rsi_15m_bullish = rsi_15m[i] > 50
        rsi_15m_bearish = rsi_15m[i] < 50
        
        new_signal = 0.0
        
        # === ENTRY CONDITIONS ===
        # Long: 4h bullish + 1h RSI pullback + MACD bullish + (volume OR EMA bullish)
        if bull_trend_4h and rsi_pullback_long and macd_bullish:
            # Need volume confirmation OR strong EMA structure
            if vol_confirmed or (ema_bullish and rsi_15m_bullish):
                new_signal = SIZE_BASE
        
        # Short: 4h bearish + 1h RSI pullback + MACD bearish + (volume OR EMA bearish)
        if bear_trend_4h and rsi_pullback_short and macd_bearish:
            # Need volume confirmation OR strong EMA structure
            if vol_confirmed or (ema_bearish and rsi_15m_bearish):
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.5 * ATR below highest close
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.5 * ATR above lowest close
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals