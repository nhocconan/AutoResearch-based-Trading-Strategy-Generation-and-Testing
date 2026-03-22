#!/usr/bin/env python3
"""
Experiment #069: 1h Keltner Breakout with 4h HMA Trend + Volume Confirmation
Hypothesis: 1h timeframe captures intraday momentum while 4h HMA filters false breakouts.
Keltner Channels (ATR-based) work better than Bollinger in crypto due to volatility clustering.
Volume spike confirmation reduces false breakouts - a key lesson from 63 failed strategies.
ADX regime filter distinguishes trending vs ranging to apply appropriate entry logic.
Why this might work: 1h is faster than 12h (more trades) but slower than 15m/30m (less noise).
4h HMA provides trend bias without excessive lag. Volume confirmation is underutilized edge.
Position sizing: 0.25 base, 0.35 strong trend + volume, discrete levels to minimize fees.
Stoploss: 2.5*ATR trailing stop to protect against crypto volatility spikes.
Timeframe: 1h (REQUIRED), HTF: 4h via mtf_data helper (call ONCE before loop).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_keltner_4h_hma_vol_adx_v1"
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

def calculate_keltner_channels(high, low, close, atr_period=14, ema_period=20, multiplier=2.0):
    """
    Calculate Keltner Channels - ATR-based volatility bands.
    Better than Bollinger for crypto due to volatility clustering.
    """
    ema = pd.Series(close).ewm(span=ema_period, min_periods=ema_period, adjust=False).mean().values
    atr = calculate_atr(high, low, close, atr_period)
    
    upper = ema + multiplier * atr
    lower = ema - multiplier * atr
    width = (upper - lower) / ema * 100
    
    return upper, lower, width, ema

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

def calculate_volume_spike(volume, period=20):
    """Calculate volume spike ratio (current volume vs rolling average)."""
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / (vol_avg + 1e-10)
    return vol_ratio

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_sma(close, period):
    """Calculate SMA."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_momentum(close, period=10):
    """Calculate rate of change momentum."""
    roc = pd.Series(close).pct_change(periods=period).values
    return roc

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
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    rsi_7 = calculate_rsi(close, 7)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    # Keltner Channels
    kc_upper, kc_lower, kc_width, kc_mid = calculate_keltner_channels(high, low, close, 14, 20, 2.0)
    
    # Volume spike detection
    vol_ratio = calculate_volume_spike(volume, 20)
    
    # Momentum
    momentum = calculate_momentum(close, 10)
    
    # KAMA for adaptive trend
    def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
        n = len(close)
        kama = np.zeros(n)
        kama[:] = np.nan
        
        change = np.abs(close - np.roll(close, er_period))
        change[:er_period] = np.nan
        
        volatility = np.zeros(n)
        for i in range(er_period, n):
            volatility[i] = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
        
        er = change / (volatility + 1e-10)
        sc = (er * (2/(fast_period+1) - 2/(slow_period+1)) + 2/(slow_period+1)) ** 2
        
        kama[er_period] = close[er_period]
        for i in range(er_period + 1, n):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        
        return kama
    
    kama = calculate_kama(close)
    
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
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = intermediate trend bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # EMA alignment on 1h
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # Price vs SMA200
        above_sma200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_sma200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # KAMA trend
        kama_bullish = not np.isnan(kama[i]) and close[i] > kama[i]
        kama_bearish = not np.isnan(kama[i]) and close[i] < kama[i]
        
        # === TREND STRENGTH / REGIME ===
        trending_regime = adx[i] > 20
        strong_trend = adx[i] > 28
        ranging_regime = adx[i] < 18
        
        # DI crossover
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = plus_di[i] < minus_di[i]
        
        # === KELTNER BREAKOUT SIGNALS ===
        # Breakout above upper band
        breakout_long = close[i] > kc_upper[i]
        # Breakout below lower band
        breakout_short = close[i] < kc_lower[i]
        
        # === VOLUME CONFIRMATION ===
        volume_spike = vol_ratio[i] > 1.5  # 50% above average
        strong_volume = vol_ratio[i] > 2.0  # 100% above average
        
        # === RSI CONDITIONS ===
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_neutral = 40 <= rsi[i] <= 60
        rsi_bullish = rsi[i] > 50
        rsi_bearish = rsi[i] < 50
        
        # === MOMENTUM ===
        momentum_positive = momentum[i] > 0.01  # >1% over 10 periods
        momentum_negative = momentum[i] < -0.01
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS (multiple paths for more trades) ===
        
        # Path 1: Keltner breakout + trend alignment + volume (trending regime)
        if trending_regime and bull_trend_4h:
            if breakout_long and di_bullish:
                if volume_spike:
                    if strong_trend:
                        new_signal = SIZE_STRONG
                    else:
                        new_signal = SIZE_BASE
                elif strong_volume:
                    new_signal = SIZE_BASE
        
        # Path 2: EMA crossover + 4h trend + RSI confirmation
        if bull_trend_4h and ema_bullish:
            if rsi_bullish and rsi[i] < 70:
                if momentum_positive:
                    new_signal = SIZE_BASE
        
        # Path 3: KAMA trend + pullback to Keltner mid
        if kama_bullish and bull_trend_4h:
            if close[i] > kc_mid[i] and close[i] < kc_upper[i]:
                if rsi[i] > 45 and rsi[i] < 65:
                    new_signal = SIZE_HALF
        
        # Path 4: Mean reversion in ranging regime (oversold bounce)
        if ranging_regime:
            if rsi_oversold and close[i] < kc_lower[i] * 1.01:
                if above_sma200:
                    new_signal = SIZE_HALF
        
        # Path 5: Simple trend continuation
        if bull_trend_4h and ema_bullish and di_bullish:
            if rsi_neutral and close[i] > ema_21[i]:
                if momentum_positive:
                    new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS (multiple paths for more trades) ===
        
        # Path 1: Keltner breakout + trend alignment + volume (trending regime)
        if trending_regime and bear_trend_4h:
            if breakout_short and di_bearish:
                if volume_spike:
                    if strong_trend:
                        new_signal = -SIZE_STRONG
                    else:
                        new_signal = -SIZE_BASE
                elif strong_volume:
                    new_signal = -SIZE_BASE
        
        # Path 2: EMA crossover + 4h trend + RSI confirmation
        if bear_trend_4h and ema_bearish:
            if rsi_bearish and rsi[i] > 30:
                if momentum_negative:
                    new_signal = -SIZE_BASE
        
        # Path 3: KAMA trend + pullback to Keltner mid
        if kama_bearish and bear_trend_4h:
            if close[i] < kc_mid[i] and close[i] > kc_lower[i]:
                if rsi[i] > 35 and rsi[i] < 55:
                    new_signal = -SIZE_HALF
        
        # Path 4: Mean reversion in ranging regime (overbought drop)
        if ranging_regime:
            if rsi_overbought and close[i] > kc_upper[i] * 0.99:
                if below_sma200:
                    new_signal = -SIZE_HALF
        
        # Path 5: Simple trend continuation
        if bear_trend_4h and ema_bearish and di_bearish:
            if rsi_neutral and close[i] < ema_21[i]:
                if momentum_negative:
                    new_signal = -SIZE_BASE
        
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