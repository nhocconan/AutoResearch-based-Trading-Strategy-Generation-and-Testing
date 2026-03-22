#!/usr/bin/env python3
"""
Experiment #062: 30m Keltner Channel Breakout with 4h HMA Trend + MACD Momentum
Hypothesis: 30m timeframe captures intraday momentum while 4h HMA filters major trend direction.
Keltner Channels (ATR-based) provide adaptive breakout levels that adjust to volatility.
MACD histogram confirms momentum direction before entry - avoids false breakouts.
ADX regime filter distinguishes trending (breakout) vs ranging (mean reversion) markets.
Why this might work: Keltner breakouts work well in trending markets, MACD filters momentum.
4h HMA provides trend bias without excessive lag. ADX prevents trading breakouts in chop.
Entry conditions designed for 10+ trades per symbol (not too strict).
Position sizing: 0.25 base, 0.35 strong trend, discrete levels to minimize fee churn.
Timeframe: 30m (REQUIRED), HTF: 4h via mtf_data helper (call ONCE before loop).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_keltner_macd_4h_hma_adx_v1"
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

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram."""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line.values, signal_line.values, histogram.values

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

def calculate_keltner_channels(high, low, close, atr_period=14, atr_mult=2.0, ema_period=20):
    """
    Calculate Keltner Channels (EMA-based middle, ATR-based bands).
    Upper = EMA(20) + 2*ATR(14)
    Lower = EMA(20) - 2*ATR(14)
    """
    ema_mid = calculate_ema(close, ema_period)
    atr = calculate_atr(high, low, close, atr_period)
    
    upper = ema_mid + atr_mult * atr
    lower = ema_mid - atr_mult * atr
    width = (upper - lower) / ema_mid * 100
    
    return upper, lower, width

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

def calculate_sma(close, period):
    """Calculate SMA."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

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
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    # Keltner Channels for breakout detection
    kc_upper, kc_lower, kc_width = calculate_keltner_channels(high, low, close, 14, 2.0, 20)
    
    # MACD for momentum confirmation
    macd_line, macd_signal, macd_hist = calculate_macd(close, 12, 26, 9)
    
    # HMA on 30m for short-term trend
    hma_30m = calculate_hma(close, 21)
    hma_30m_fast = calculate_hma(close, 10)
    
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
        
        if np.isnan(rsi[i]) or np.isnan(ema_21[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(macd_hist[i]) or np.isnan(macd_line[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = intermediate trend bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # 30m HMA = short-term trend
        bull_trend_30m = hma_30m_fast[i] > hma_30m[i] if not np.isnan(hma_30m_fast[i]) else False
        bear_trend_30m = hma_30m_fast[i] < hma_30m[i] if not np.isnan(hma_30m_fast[i]) else False
        
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
        
        # === KELTNER CHANNEL BREAKOUT SIGNALS ===
        # Breakout above upper band
        kc_breakout_long = close[i] > kc_upper[i]
        # Breakout below lower band
        kc_breakout_short = close[i] < kc_lower[i]
        # Pullback to middle (EMA)
        kc_pullback_long = close[i] < kc_upper[i] and close[i] > ema_21[i] and close[i] < ema_21[i] * 1.01
        kc_pullback_short = close[i] > kc_lower[i] and close[i] < ema_21[i] and close[i] > ema_21[i] * 0.99
        
        # === MACD MOMENTUM CONFIRMATION ===
        macd_bullish = macd_hist[i] > 0 and macd_line[i] > macd_signal[i]
        macd_bearish = macd_hist[i] < 0 and macd_line[i] < macd_signal[i]
        macd_cross_up = macd_hist[i] > 0 and macd_hist[i-1] <= 0 if i > 0 else False
        macd_cross_down = macd_hist[i] < 0 and macd_hist[i-1] >= 0 if i > 0 else False
        
        # === RSI CONDITIONS ===
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_neutral = 40 <= rsi[i] <= 60
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS (multiple paths for more trades) ===
        
        # Path 1: Keltner breakout + trend alignment + MACD confirmation (trending)
        if trending_regime and bull_trend_4h:
            if kc_breakout_long and macd_bullish and di_bullish:
                if strong_trend:
                    new_signal = SIZE_STRONG
                else:
                    new_signal = SIZE_BASE
        
        # Path 2: MACD cross + HMA trend + RSI confirmation
        if bull_trend_4h and bull_trend_30m:
            if macd_cross_up and rsi[i] > 40 and rsi[i] < 70:
                if ema_bullish:
                    new_signal = SIZE_BASE
        
        # Path 3: Keltner pullback to EMA in uptrend
        if bull_trend_4h and ema_bullish:
            if close[i] <= ema_21[i] * 1.005 and close[i] >= ema_21[i] * 0.995:
                if rsi[i] > 35 and rsi[i] < 60:
                    if macd_hist[i] > -50:  # MACD not too bearish
                        new_signal = SIZE_HALF
        
        # Path 4: RSI oversold bounce in uptrend
        if bull_trend_4h and above_sma200:
            if rsi_oversold and rsi[i] > rsi[i-1] if i > 0 else False:
                if macd_hist[i] > macd_hist[i-1] if i > 0 else False:
                    new_signal = SIZE_HALF
        
        # === SHORT ENTRY CONDITIONS (multiple paths for more trades) ===
        
        # Path 1: Keltner breakout + trend alignment + MACD confirmation (trending)
        if trending_regime and bear_trend_4h:
            if kc_breakout_short and macd_bearish and di_bearish:
                if strong_trend:
                    new_signal = -SIZE_STRONG
                else:
                    new_signal = -SIZE_BASE
        
        # Path 2: MACD cross + HMA trend + RSI confirmation
        if bear_trend_4h and bear_trend_30m:
            if macd_cross_down and rsi[i] > 30 and rsi[i] < 60:
                if ema_bearish:
                    new_signal = -SIZE_BASE
        
        # Path 3: Keltner pullback to EMA in downtrend
        if bear_trend_4h and ema_bearish:
            if close[i] >= ema_21[i] * 0.995 and close[i] <= ema_21[i] * 1.005:
                if rsi[i] > 40 and rsi[i] < 65:
                    if macd_hist[i] < 50:  # MACD not too bullish
                        new_signal = -SIZE_HALF
        
        # Path 4: RSI overbought rejection in downtrend
        if bear_trend_4h and below_sma200:
            if rsi_overbought and rsi[i] < rsi[i-1] if i > 0 else False:
                if macd_hist[i] < macd_hist[i-1] if i > 0 else False:
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