#!/usr/bin/env python3
"""
Experiment #051: 1h Multi-Factor Trend Following with 4h HMA Bias
Hypothesis: 1h timeframe balances signal frequency and noise better than 15m/30m (failed) or 1d (too few trades).
Key insight: Combine 4h HMA trend bias + 1h RSI pullback + volume confirmation + asymmetric sizing.
Why this might work: 
- 4h HMA provides stable trend direction (proven in baseline)
- RSI(7) pullback entries catch dips in uptrends without waiting for extremes
- Volume confirmation filters false breakouts (taker_buy_ratio > 0.55)
- Asymmetric sizing: larger longs in bull, smaller shorts in bull (reduces bear trap losses)
- ATR stoploss at 2.5*ATR protects capital
Position sizing: 0.30 base, 0.35 strong trend, 0.20 counter-trend. Discrete levels only.
Must generate 10+ trades: entry conditions loosened (RSI 35-65 range, not extremes).
Timeframe: 1h (REQUIRED for exp#051), HTF: 4h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_multifactor_4h_hma_volume_v1"
timeframe = "1h"
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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bb_width = (upper - lower) / (sma + 1e-10)
    return upper, lower, bb_width

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average."""
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Efficiency Ratio
    change = np.abs(close - np.roll(close, er_period))
    volatility = np.zeros(n)
    for i in range(er_period, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
    
    er = np.zeros(n)
    mask = volatility > 0
    er[mask] = change[mask] / volatility[mask]
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_supertrend(high, low, close, period=10, mult=3.0):
    """Calculate Supertrend indicator."""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    hl2 = (high + low) / 2
    upper_band = hl2 + mult * atr
    lower_band = hl2 - mult * atr
    
    supertrend = np.zeros(n)
    supertrend[:] = np.nan
    direction = np.ones(n)  # 1 = up, -1 = down
    
    supertrend[period] = lower_band[period]
    for i in range(period + 1, n):
        if close[i-1] > supertrend[i-1]:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
            direction[i] = 1
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
            direction[i] = -1
    
    return supertrend, direction

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi_7 = calculate_rsi(close, 7)
    rsi_14 = calculate_rsi(close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    ema_200 = calculate_ema(close, 200)
    sma_200 = calculate_sma(close, 200)
    kama = calculate_kama(close, 10, 2, 30)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    bb_upper, bb_lower, bb_width = calculate_bollinger(close, 20, 2.0)
    
    # Volume ratio (taker buy / total volume)
    taker_ratio = np.zeros(n)
    mask = volume > 0
    taker_ratio[mask] = taker_buy_volume[mask] / volume[mask]
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.30
    SIZE_STRONG = 0.35
    SIZE_WEAK = 0.20
    
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
        
        if np.isnan(rsi_7[i]) or np.isnan(ema_21[i]) or np.isnan(supertrend[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF) - main regime filter
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # 1h trend confirmation
        bull_trend_1h = close[i] > ema_50[i] and ema_21[i] > ema_50[i]
        bear_trend_1h = close[i] < ema_50[i] and ema_21[i] < ema_50[i]
        
        # Long-term trend filter
        above_200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # Supertrend direction
        st_bullish = st_direction[i] == 1
        st_bearish = st_direction[i] == -1
        
        # RSI pullback levels (not extremes - want more trades)
        rsi_pullback_long = 35 <= rsi_7[i] <= 55
        rsi_pullback_short = 45 <= rsi_7[i] <= 65
        
        # RSI momentum
        rsi_momentum_long = rsi_7[i] > 50
        rsi_momentum_short = rsi_7[i] < 50
        
        # Volume confirmation
        volume_bullish = taker_ratio[i] > 0.55
        volume_bearish = taker_ratio[i] < 0.45
        
        # Price position relative to EMA21
        price_above_ema21 = close[i] > ema_21[i]
        price_below_ema21 = close[i] < ema_21[i]
        
        # Price pullback to EMA21 (within 3%)
        price_near_ema21_long = close[i] <= ema_21[i] * 1.02 and close[i] >= ema_21[i] * 0.98
        price_near_ema21_short = close[i] >= ema_21[i] * 0.98 and close[i] <= ema_21[i] * 1.02
        
        # Bollinger Band signals
        price_at_lower_bb = close[i] <= bb_lower[i] * 1.01
        price_at_upper_bb = close[i] >= bb_upper[i] * 0.99
        
        # KAMA trend
        kama_bullish = close[i] > kama[i]
        kama_bearish = close[i] < kama[i]
        
        new_signal = 0.0
        
        # === PRIMARY LONG SIGNALS (4h bull + 1h pullback) ===
        if bull_trend_4h:
            # Strong long: all conditions align
            if bull_trend_1h and st_bullish and rsi_pullback_long:
                if price_near_ema21_long and volume_bullish:
                    new_signal = SIZE_STRONG
                elif price_at_lower_bb and kama_bullish:
                    new_signal = SIZE_BASE
            
            # Moderate long: trend + RSI momentum
            elif above_200 and rsi_momentum_long:
                if price_above_ema21 and volume_bullish:
                    new_signal = SIZE_BASE
            
            # Weak long: counter-trend bounce (smaller size)
            elif rsi_7[i] < 30 and price_at_lower_bb:
                if bull_trend_4h:  # only counter-trend in 4h bull
                    new_signal = SIZE_WEAK
        
        # === PRIMARY SHORT SIGNALS (4h bear + 1h bounce) ===
        elif bear_trend_4h:
            # Strong short: all conditions align
            if bear_trend_1h and st_bearish and rsi_pullback_short:
                if price_near_ema21_short and volume_bearish:
                    new_signal = -SIZE_STRONG
                elif price_at_upper_bb and kama_bearish:
                    new_signal = -SIZE_BASE
            
            # Moderate short: trend + RSI momentum
            elif below_200 and rsi_momentum_short:
                if price_below_ema21 and volume_bearish:
                    new_signal = -SIZE_BASE
            
            # Weak short: counter-trend bounce (smaller size)
            elif rsi_7[i] > 70 and price_at_upper_bb:
                if bear_trend_4h:  # only counter-trend in 4h bear
                    new_signal = -SIZE_WEAK
        
        # === TRANSITION/NEUTRAL (4h unclear) ===
        else:
            # Only take very strong signals in neutral regime
            if rsi_7[i] < 25 and price_at_lower_bb and above_200:
                new_signal = SIZE_WEAK
            elif rsi_7[i] > 75 and price_at_upper_bb and below_200:
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