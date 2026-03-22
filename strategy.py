#!/usr/bin/env python3
"""
Experiment #032: 30m Trend-Follow with 4h HMA Bias + RSI Pullback
Hypothesis: 30m captures intraday trends while 4h HMA provides cleaner regime filter than EMA.
Key insight: Previous 30m strategies failed due to too many conflicting filters (0 trades) or 
overly complex regime detection. This uses simpler logic: 4h HMA for trend bias, 30m EMA 
pullback entries with loose RSI filter, ATR stops. Fewer conditions = more trades generated.
Position sizing: 0.25 discrete levels, stoploss at 2.5*ATR to control drawdown.
Timeframe: 30m (REQUIRED for exp#032), HTF: 4h via mtf_data helper.
Why this might work: 30m has good balance between noise (5m/15m) and lag (1h/4h). 
4h HMA smoother than 4h EMA for regime detection. Looser RSI (35-65) ensures trades generate.
Must generate 10+ trades on train, 3+ on test - entry conditions deliberately loosened.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_trend_4h_hma_rsi_pullback_v1"
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

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_sma(close, period):
    """Calculate SMA."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_keltner(high, low, close, ema_period=20, atr_period=10, mult=2.0):
    """Calculate Keltner Channels for volatility-based entries."""
    ema = calculate_ema(close, ema_period)
    atr = calculate_atr(high, low, close, atr_period)
    upper = ema + mult * atr
    lower = ema - mult * atr
    return upper, lower, ema

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
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
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    ema_200 = calculate_ema(close, 200)
    sma_200 = calculate_sma(close, 200)
    
    # Keltner Channels for volatility entries
    keltner_upper, keltner_lower, keltner_mid = calculate_keltner(high, low, close, 20, 10, 2.0)
    
    # HMA on 30m for faster trend signal
    hma_30m = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_HALF = 0.125
    
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
        
        if np.isnan(rsi[i]) or np.isnan(ema_21[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF) - main regime filter
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # 30m trend confirmation - LOOSENED for more trades
        bull_trend_30m = ema_21[i] > ema_50[i]
        bear_trend_30m = ema_21[i] < ema_50[i]
        
        # Long-term trend filter
        above_200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # RSI conditions - LOOSENED significantly for more trades
        rsi_bullish = rsi[i] > 40  # Much looser than before
        rsi_bearish = rsi[i] < 60  # Much looser than before
        rsi_neutral = 35 < rsi[i] < 65  # Wide neutral zone
        
        # HMA crossover on 30m - primary entry signal
        hma_cross_long = False
        hma_cross_short = False
        if i >= 1 and not np.isnan(hma_30m[i]) and not np.isnan(hma_30m[i-1]):
            hma_cross_long = hma_30m[i] > ema_50[i] and hma_30m[i-1] <= ema_50[i-1]
            hma_cross_short = hma_30m[i] < ema_50[i] and hma_30m[i-1] >= ema_50[i-1]
        
        # Price pullback to EMA21 - secondary entry
        price_near_ema21_long = close[i] <= ema_21[i] * 1.015 and close[i] >= ema_21[i] * 0.985
        price_near_ema21_short = close[i] >= ema_21[i] * 0.985 and close[i] <= ema_21[i] * 1.015
        
        # Keltner breakout - tertiary entry
        keltner_breakout_long = close[i] > keltner_upper[i]
        keltner_breakout_short = close[i] < keltner_lower[i]
        
        # Price action: higher low for long, lower high for short
        higher_low = False
        lower_high = False
        if i >= 5:
            higher_low = low[i] > min(low[i-3:i])
            lower_high = high[i] < max(high[i-3:i])
        
        new_signal = 0.0
        
        # === LONG ENTRIES (when 4h bullish) ===
        if bull_trend_4h:
            # Primary: HMA crossover with trend alignment
            if hma_cross_long and bull_trend_30m and rsi_bullish:
                new_signal = SIZE_BASE
            
            # Secondary: Pullback to EMA21 in uptrend
            elif price_near_ema21_long and bull_trend_30m and above_200:
                new_signal = SIZE_BASE
            
            # Tertiary: Keltner breakout with momentum
            elif keltner_breakout_long and bull_trend_30m and rsi[i] > 45:
                new_signal = SIZE_HALF
            
            # Momentum: Higher low pattern
            elif higher_low and bull_trend_30m and rsi[i] > 40:
                new_signal = SIZE_HALF
        
        # === SHORT ENTRIES (when 4h bearish) ===
        elif bear_trend_4h:
            # Primary: HMA crossover with trend alignment
            if hma_cross_short and bear_trend_30m and rsi_bearish:
                new_signal = -SIZE_BASE
            
            # Secondary: Bounce to EMA21 in downtrend
            elif price_near_ema21_short and bear_trend_30m and below_200:
                new_signal = -SIZE_BASE
            
            # Tertiary: Keltner breakdown with momentum
            elif keltner_breakout_short and bear_trend_30m and rsi[i] < 55:
                new_signal = -SIZE_HALF
            
            # Momentum: Lower high pattern
            elif lower_high and bear_trend_30m and rsi[i] < 60:
                new_signal = -SIZE_HALF
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            if close[i] < trailing_stop and entry_price > 0:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0:
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            if close[i] > trailing_stop and entry_price > 0:
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