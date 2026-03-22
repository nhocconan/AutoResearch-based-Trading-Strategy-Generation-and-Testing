#!/usr/bin/env python3
"""
Experiment #061: 15m RSI Pullback with 4h HMA Trend + 1h Momentum Filter
Hypothesis: 15m timeframe captures intraday momentum while 4h HMA provides trend bias.
Key insight: RSI pullbacks in trending markets have high win rates. 1h EMA confirms momentum.
Why this might work: Fast TF entries (15m) with slow TF filter (4h) reduces whipsaws.
RSI(7) pullback to 40-50 in uptrend catches continuations. Volume filter confirms breakout.
Position sizing: 0.25 base, 0.30 strong trend, discrete levels to minimize fee churn.
Timeframe: 15m (REQUIRED), HTF: 4h + 1h via mtf_data helper (call ONCE before loop).
Entry conditions loosened to ensure 10+ trades per symbol on all BTC/ETH/SOL.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi_pullback_4h_hma_1h_ema_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

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
    """Calculate RSI."""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[~mask] = 100.0
    
    return rsi

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_sma(close, period):
    """Calculate SMA."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD histogram."""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line.values, signal_line.values, histogram.values

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs SMA."""
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    ratio = volume / (vol_sma + 1e-10)
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    ema_1h_21 = calculate_ema(df_1h['close'].values, 21)
    ema_1h_50 = calculate_ema(df_1h['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    ema_1h_21_aligned = align_htf_to_ltf(prices, df_1h, ema_1h_21)
    ema_1h_50_aligned = align_htf_to_ltf(prices, df_1h, ema_1h_50)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi_7 = calculate_rsi(close, 7)
    rsi_14 = calculate_rsi(close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    sma_200 = calculate_sma(close, 200)
    macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    # Bollinger Bands for mean reversion context
    bb_sma = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_sma + 2.0 * bb_std
    bb_lower = bb_sma - 2.0 * bb_std
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    SIZE_WEAK = 0.20
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(ema_21[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = primary trend bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # 1h EMA crossover = intermediate momentum
        ema_1h_bullish = not np.isnan(ema_1h_21_aligned[i]) and ema_1h_21_aligned[i] > ema_1h_50_aligned[i]
        ema_1h_bearish = not np.isnan(ema_1h_21_aligned[i]) and ema_1h_21_aligned[i] < ema_1h_50_aligned[i]
        
        # 15m EMA alignment
        ema_15m_bullish = ema_21[i] > ema_50[i]
        ema_15m_bearish = ema_21[i] < ema_50[i]
        
        # Price vs SMA200
        above_sma200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_sma200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # === RSI PULLBACK CONDITIONS ===
        # In uptrend: RSI pulls back to 40-55 zone
        rsi_pullback_long = 40 <= rsi_7[i] <= 55
        # In downtrend: RSI pulls back to 45-60 zone
        rsi_pullback_short = 45 <= rsi_7[i] <= 60
        
        # RSI momentum confirmation
        rsi_momentum_long = rsi_7[i] > rsi_14[i]
        rsi_momentum_short = rsi_7[i] < rsi_14[i]
        
        # === MACD CONFIRMATION ===
        macd_bullish = macd_hist[i] > 0
        macd_bearish = macd_hist[i] < 0
        macd_cross_up = i > 0 and macd_hist[i] > 0 and macd_hist[i-1] <= 0
        macd_cross_down = i > 0 and macd_hist[i] < 0 and macd_hist[i-1] >= 0
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = vol_ratio[i] > 1.2
        
        # === BOLLINGER BAND POSITION ===
        near_bb_lower = close[i] <= bb_lower[i] * 1.01 if not np.isnan(bb_lower[i]) else False
        near_bb_upper = close[i] >= bb_upper[i] * 0.99 if not np.isnan(bb_upper[i]) else False
        
        # === TREND STRENGTH ===
        strong_trend_long = bull_trend_4h and ema_1h_bullish and ema_15m_bullish
        strong_trend_short = bear_trend_4h and ema_1h_bearish and ema_15m_bearish
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS (multiple paths for more trades) ===
        
        # Path 1: Strong trend + RSI pullback + MACD confirm
        if strong_trend_long:
            if rsi_pullback_long and macd_bullish:
                if volume_confirmed:
                    new_signal = SIZE_STRONG
                else:
                    new_signal = SIZE_BASE
        
        # Path 2: 4h trend + 15m EMA + RSI momentum
        if bull_trend_4h and ema_15m_bullish:
            if rsi_7[i] > 45 and rsi_7[i] < 60:
                if rsi_momentum_long and above_sma200:
                    new_signal = SIZE_BASE
        
        # Path 3: MACD cross + trend alignment
        if bull_trend_4h:
            if macd_cross_up and rsi_7[i] > 50:
                new_signal = SIZE_BASE
        
        # Path 4: Simple trend continuation (looser for more trades)
        if bull_trend_4h and ema_1h_bullish:
            if rsi_7[i] > 50 and rsi_7[i] < 70:
                if close[i] > ema_21[i]:
                    new_signal = SIZE_WEAK
        
        # Path 5: BB mean reversion in uptrend
        if bull_trend_4h:
            if near_bb_lower and rsi_7[i] < 45:
                new_signal = SIZE_WEAK
        
        # === SHORT ENTRY CONDITIONS (multiple paths for more trades) ===
        
        # Path 1: Strong trend + RSI pullback + MACD confirm
        if strong_trend_short:
            if rsi_pullback_short and macd_bearish:
                if volume_confirmed:
                    new_signal = -SIZE_STRONG
                else:
                    new_signal = -SIZE_BASE
        
        # Path 2: 4h trend + 15m EMA + RSI momentum
        if bear_trend_4h and ema_15m_bearish:
            if rsi_7[i] > 40 and rsi_7[i] < 55:
                if rsi_momentum_short and below_sma200:
                    new_signal = -SIZE_BASE
        
        # Path 3: MACD cross + trend alignment
        if bear_trend_4h:
            if macd_cross_down and rsi_7[i] < 50:
                new_signal = -SIZE_BASE
        
        # Path 4: Simple trend continuation (looser for more trades)
        if bear_trend_4h and ema_1h_bearish:
            if rsi_7[i] > 30 and rsi_7[i] < 50:
                if close[i] < ema_21[i]:
                    new_signal = -SIZE_WEAK
        
        # Path 5: BB mean reversion in downtrend
        if bear_trend_4h:
            if near_bb_upper and rsi_7[i] > 55:
                new_signal = -SIZE_WEAK
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals