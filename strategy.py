#!/usr/bin/env python3
"""
Experiment #008: 30m Asymmetric Trend-Follow with 4h HMA Bias
Hypothesis: Simple asymmetric logic works better than complex regime detection.
Long only when 4h HMA bullish + price pullback to EMA21. Short only when 4h HMA bearish + price bounce to EMA21.
This avoids the whipsaw that destroyed strategy #002 (complex ADX/BB regime failed with Sharpe=-1.428).
Key insight: BTC/ETH 2022 crash and 2025 bear market need short bias, but 2021 bull needs long bias.
4h HMA provides the regime filter without overfitting. RSI confirms pullback/bounce exhaustion.
Position sizing: 0.30 discrete, stoploss at 2.5*ATR trailing. Fewer but higher quality trades.
Timeframe: 30m (REQUIRED), HTF: 4h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_asymmetric_4h_hma_rsi_v1"
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
    """Calculate Keltner Channel for volatility-based entries."""
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
    
    # Keltner Channel for volatility entries
    keltner_upper, keltner_lower, keltner_mid = calculate_keltner(high, low, close, 20, 10, 2.0)
    
    # HMA on 30m for faster trend
    hma_30m = calculate_hma(close, 21)
    
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
        
        if np.isnan(rsi[i]) or np.isnan(ema_21[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF) - this is the main regime filter
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # 30m trend confirmation
        bull_trend_30m = close[i] > ema_50[i] and ema_21[i] > ema_50[i]
        bear_trend_30m = close[i] < ema_50[i] and ema_21[i] < ema_50[i]
        
        # Long-term trend filter (only trade with 200 EMA direction)
        above_200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else True
        below_200 = close[i] < sma_200[i] if not np.isnan(sma_200[i]) else False
        
        # RSI pullback/bounce conditions
        rsi_pullback_long = 35 < rsi[i] < 55  # Pullback in uptrend
        rsi_bounce_short = 45 < rsi[i] < 65   # Bounce in downtrend
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        
        # Keltner Channel entries
        price_near_lower_keltner = close[i] < keltner_lower[i] * 1.005
        price_near_upper_keltner = close[i] > keltner_upper[i] * 0.995
        price_at_mid_keltner = abs(close[i] - keltner_mid[i]) < atr[i] * 0.5
        
        # HMA crossover on 30m
        hma_cross_long = hma_30m[i] > ema_50[i] and hma_30m[i-1] <= ema_50[i-1] if i >= 1 else False
        hma_cross_short = hma_30m[i] < ema_50[i] and hma_30m[i-1] >= ema_50[i-1] if i >= 1 else False
        
        # Price action: higher low for long, lower high for short
        higher_low = low[i] > low[i-3] if i >= 3 else False
        lower_high = high[i] < high[i-3] if i >= 3 else False
        
        new_signal = 0.0
        
        # === LONG ENTRIES (only when 4h bullish) ===
        if bull_trend_4h:
            # Primary: Pullback to EMA21 with RSI confirmation
            if (close[i] <= ema_21[i] * 1.01 and close[i] >= ema_21[i] * 0.99 and 
                rsi_pullback_long and above_200):
                new_signal = SIZE_BASE
            
            # Secondary: Keltner lower touch with RSI oversold
            elif price_near_lower_keltner and rsi_oversold and bull_trend_30m:
                new_signal = SIZE_BASE
            
            # Tertiary: HMA crossover with 4h confirmation
            elif hma_cross_long and bull_trend_4h and above_200:
                new_signal = SIZE_HALF
            
            # Momentum continuation: RSI rising from oversold
            elif rsi_oversold and rsi[i] > rsi[i-2] if i >= 2 else False and bull_trend_30m:
                new_signal = SIZE_HALF
        
        # === SHORT ENTRIES (only when 4h bearish) ===
        elif bear_trend_4h:
            # Primary: Bounce to EMA21 with RSI confirmation
            if (close[i] >= ema_21[i] * 0.99 and close[i] <= ema_21[i] * 1.01 and 
                rsi_bounce_short and below_200):
                new_signal = -SIZE_BASE
            
            # Secondary: Keltner upper touch with RSI overbought
            elif price_near_upper_keltner and rsi_overbought and bear_trend_30m:
                new_signal = -SIZE_BASE
            
            # Tertiary: HMA crossover with 4h confirmation
            elif hma_cross_short and bear_trend_4h and below_200:
                new_signal = -SIZE_HALF
            
            # Momentum continuation: RSI falling from overbought
            elif rsi_overbought and rsi[i] < rsi[i-2] if i >= 2 else False and bear_trend_30m:
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