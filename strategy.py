#!/usr/bin/env python3
"""
Experiment #014: 30m Multi-Timeframe Momentum with 4h HMA Regime Filter
Hypothesis: 30m captures intraday momentum swings while 4h HMA provides stable regime bias.
Key insight: Previous 30m strategies failed due to too-strict filters (vol breakout = 0 trades) 
or wrong signal type (funding contrarian = -3.4 Sharpe). This uses MACD momentum + RSI pullback
with LOOSE entry conditions to ensure 10+ trades. 4h HMA smoother than 4h EMA for regime.
Timeframe: 30m (REQUIRED for exp#014), HTF: 4h via mtf_data helper.
Position sizing: 0.30 base, 0.15 half-size for weaker signals. Stoploss at 2.5*ATR.
Why this might work: MACD histogram captures momentum shifts better than EMA crossover.
RSI 35-65 range (not 30-70) generates more signals. Volume filter is lenient (1.2x avg).
Must generate 10+ trades on train, 3+ on test - conditions deliberately loosened.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_macd_rsi_4h_hma_v1"
timeframe = "30m"
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

def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram."""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line.values, signal_line.values, histogram.values

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_sma(close, period):
    """Calculate SMA."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average."""
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / (vol_sma + 1e-10)
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    ema_200 = calculate_ema(close, 200)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.30
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(macd_hist[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF) - main regime filter
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # 30m trend confirmation
        bull_trend_30m = close[i] > ema_50[i] and ema_21[i] > ema_50[i]
        bear_trend_30m = close[i] < ema_50[i] and ema_21[i] < ema_50[i]
        
        # Long-term trend filter
        above_200 = not np.isnan(ema_200[i]) and close[i] > ema_200[i]
        below_200 = not np.isnan(ema_200[i]) and close[i] < ema_200[i]
        
        # RSI conditions - LOOSENED for more trades (35-65 instead of 30-70)
        rsi_pullback_long = 35 < rsi[i] < 60
        rsi_bounce_short = 40 < rsi[i] < 65
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        
        # MACD momentum
        macd_bullish = macd_hist[i] > 0 and macd_hist[i] > macd_hist[i-1] if i > 0 else False
        macd_bearish = macd_hist[i] < 0 and macd_hist[i] < macd_hist[i-1] if i > 0 else False
        macd_cross_up = macd_line[i] > macd_signal[i] and macd_line[i-1] <= macd_signal[i-1] if i > 0 else False
        macd_cross_down = macd_line[i] < macd_signal[i] and macd_line[i-1] >= macd_signal[i-1] if i > 0 else False
        
        # Volume confirmation - lenient (1.2x instead of 1.5x)
        vol_confirmed = vol_ratio[i] > 1.2
        
        # Bollinger position
        near_bb_lower = close[i] <= bb_lower[i] * 1.01
        near_bb_upper = close[i] >= bb_upper[i] * 0.99
        bb_squeeze = (bb_upper[i] - bb_lower[i]) / (bb_mid[i] + 1e-10) < 0.05
        
        # Price pullback to EMA21
        price_near_ema21_long = close[i] <= ema_21[i] * 1.015 and close[i] >= ema_21[i] * 0.985
        price_near_ema21_short = close[i] >= ema_21[i] * 0.985 and close[i] <= ema_21[i] * 1.015
        
        # Price action: higher low for long, lower high for short
        higher_low = False
        lower_high = False
        if i >= 3:
            higher_low = low[i] > low[i-3]
            lower_high = high[i] < high[i-3]
        
        new_signal = 0.0
        
        # === LONG ENTRIES (only when 4h bullish) ===
        if bull_trend_4h:
            # Primary: MACD bullish + RSI pullback + volume
            if macd_bullish and rsi_pullback_long and vol_confirmed:
                new_signal = SIZE_BASE
            
            # Secondary: Price pullback to EMA21 in uptrend
            elif price_near_ema21_long and bull_trend_30m and above_200:
                new_signal = SIZE_BASE
            
            # Tertiary: RSI oversold bounce with MACD support
            elif rsi_oversold and macd_hist[i] > macd_hist[i-2] if i >= 2 else False:
                new_signal = SIZE_HALF
            
            # Momentum: MACD cross up with trend
            elif macd_cross_up and bull_trend_30m:
                new_signal = SIZE_HALF
            
            # BB mean reversion: touch lower band in uptrend
            elif near_bb_lower and bull_trend_4h and rsi[i] < 45:
                new_signal = SIZE_HALF
        
        # === SHORT ENTRIES (only when 4h bearish) ===
        elif bear_trend_4h:
            # Primary: MACD bearish + RSI bounce + volume
            if macd_bearish and rsi_bounce_short and vol_confirmed:
                new_signal = -SIZE_BASE
            
            # Secondary: Price bounce to EMA21 in downtrend
            elif price_near_ema21_short and bear_trend_30m and below_200:
                new_signal = -SIZE_BASE
            
            # Tertiary: RSI overbought rejection with MACD support
            elif rsi_overbought and macd_hist[i] < macd_hist[i-2] if i >= 2 else False:
                new_signal = -SIZE_HALF
            
            # Momentum: MACD cross down with trend
            elif macd_cross_down and bear_trend_30m:
                new_signal = -SIZE_HALF
            
            # BB mean reversion: touch upper band in downtrend
            elif near_bb_upper and bear_trend_4h and rsi[i] > 55:
                new_signal = -SIZE_HALF
        
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