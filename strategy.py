#!/usr/bin/env python3
"""
Experiment #074: 30m KAMA Adaptive Trend with 4h HMA Filter + RSI Pullback
Hypothesis: 30m timeframe captures intraday momentum while 4h HMA provides trend bias.
KAMA (Kaufman Adaptive MA) adapts to volatility - faster in trends, slower in chop.
This should outperform static EMA/HMA by reducing whipsaw in ranging markets.
RSI pullback (40-60 range) entries avoid extreme overbought/oversold traps.
Simple ATR stoploss (2.5x) protects capital during reversals.
Why this might work: KAMA's adaptive nature + HTF trend filter + moderate RSI = more trades with better timing.
Position sizing: 0.25 base, 0.35 strong trend confirmation, discrete levels.
Timeframe: 30m (REQUIRED), HTF: 4h via mtf_data helper (call ONCE before loop).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_kama_4h_hma_rsi_pullback_v1"
timeframe = "30m"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise - moves fast in trends, slow in chop.
    Efficiency Ratio (ER) determines smoothing constant.
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    if n < period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio
    change = np.abs(close - np.roll(close, period))
    change[:period] = np.nan
    
    volatility = np.zeros(n)
    for i in range(period, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-period:i+1])))
    
    er = change / (volatility + 1e-10)
    er[:period] = np.nan
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # KAMA calculation
    kama[period] = close[period]  # Initialize with price
    
    for i in range(period + 1, n):
        if np.isnan(er[i]):
            kama[i] = kama[i-1]
        else:
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    n = len(close)
    
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_momentum(close, period=10):
    """Calculate simple momentum (ROC)."""
    return (close - np.roll(close, period)) / (np.roll(close, period) + 1e-10) * 100

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
    rsi_7 = calculate_rsi(close, 7)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    
    # KAMA - adaptive trend
    kama = calculate_kama(close, 10, 2, 30)
    kama_fast = calculate_kama(close, 5, 2, 20)
    
    # EMA for confirmation
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    # Momentum
    mom_10 = calculate_momentum(close, 10)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    SIZE_HALF = 0.15
    
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
        
        if np.isnan(rsi[i]) or np.isnan(ema_21[i]) or np.isnan(kama[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = intermediate trend bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # 30m KAMA = short-term trend
        kama_bullish = not np.isnan(kama_fast[i]) and kama_fast[i] > kama[i]
        kama_bearish = not np.isnan(kama_fast[i]) and kama_fast[i] < kama[i]
        
        # Price vs KAMA
        above_kama = close[i] > kama[i]
        below_kama = close[i] < kama[i]
        
        # EMA alignment
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # Price vs SMA200
        above_sma200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_sma200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # === TREND STRENGTH / REGIME ===
        trending_regime = adx[i] > 20
        strong_trend = adx[i] > 28
        ranging_regime = adx[i] < 18
        
        # DI crossover
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = plus_di[i] < minus_di[i]
        
        # Momentum
        mom_positive = mom_10[i] > 0
        mom_negative = mom_10[i] < 0
        mom_strong_pos = mom_10[i] > 2.0
        mom_strong_neg = mom_10[i] < -2.0
        
        # === RSI PULLBACK CONDITIONS (moderate, not extreme) ===
        rsi_pullback_long = 40 <= rsi[i] <= 55
        rsi_pullback_short = 45 <= rsi[i] <= 60
        rsi_bullish = rsi[i] > 50
        rsi_bearish = rsi[i] < 50
        
        # Fast RSI confirmation
        rsi7_bullish = rsi_7[i] > 50
        rsi7_bearish = rsi_7[i] < 50
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS (multiple paths for more trades) ===
        
        # Path 1: KAMA crossover + 4h trend + RSI pullback
        if bull_trend_4h and kama_bullish:
            if rsi_pullback_long and di_bullish:
                if mom_positive:
                    new_signal = SIZE_BASE
        
        # Path 2: Strong trend continuation
        if bull_trend_4h and strong_trend:
            if above_kama and ema_bullish:
                if rsi_bullish and mom_strong_pos:
                    new_signal = SIZE_STRONG
        
        # Path 3: KAMA fast crossover + trend alignment
        if bull_trend_4h:
            if kama_bullish and ema_bullish:
                if rsi7_bullish and above_sma200:
                    new_signal = SIZE_BASE
        
        # Path 4: Trending regime breakout
        if trending_regime and bull_trend_4h:
            if close[i] > ema_21[i] and di_bullish:
                if mom_10[i] > 1.0:
                    new_signal = SIZE_BASE
        
        # Path 5: Ranging regime mean reversion (with HTF bias)
        if ranging_regime and bull_trend_4h:
            if rsi[i] < 45 and close[i] > kama[i]:
                new_signal = SIZE_HALF
        
        # === SHORT ENTRY CONDITIONS (multiple paths for more trades) ===
        
        # Path 1: KAMA crossover + 4h trend + RSI pullback
        if bear_trend_4h and kama_bearish:
            if rsi_pullback_short and di_bearish:
                if mom_negative:
                    new_signal = -SIZE_BASE
        
        # Path 2: Strong trend continuation
        if bear_trend_4h and strong_trend:
            if below_kama and ema_bearish:
                if rsi_bearish and mom_strong_neg:
                    new_signal = -SIZE_STRONG
        
        # Path 3: KAMA fast crossover + trend alignment
        if bear_trend_4h:
            if kama_bearish and ema_bearish:
                if rsi7_bearish and below_sma200:
                    new_signal = -SIZE_BASE
        
        # Path 4: Trending regime breakout
        if trending_regime and bear_trend_4h:
            if close[i] < ema_21[i] and di_bearish:
                if mom_10[i] < -1.0:
                    new_signal = -SIZE_BASE
        
        # Path 5: Ranging regime mean reversion (with HTF bias)
        if ranging_regime and bear_trend_4h:
            if rsi[i] > 55 and close[i] < kama[i]:
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