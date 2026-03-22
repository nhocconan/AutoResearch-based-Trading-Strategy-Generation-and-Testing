#!/usr/bin/env python3
"""
Experiment #067: 15m Supertrend + RSI Pullback with 4h HMA Trend Filter + 1h ADX Regime
Hypothesis: 15m timeframe needs STRONGER HTF filters than 12h due to higher noise.
Key insight: From #059 success - multi-path entries (trend + mean reversion) ensure trade frequency.
Adaptation for 15m: Use BOTH 1h AND 4h HTF filters, wider ATR stops (3.0x), lower position size (0.25).
Supertrend(10,3) for momentum direction, RSI(14) pullback for entry timing.
4h HMA(21) = primary trend bias, 1h ADX(14) = regime filter (trend vs range).
Multiple entry paths: (1) Supertrend flip + HTF alignment, (2) RSI pullback in trend, (3) Mean reversion in range.
Why this might work: Supertrend proven on crypto 15m, HTF filters reduce false signals, multi-path ensures 10+ trades.
Position sizing: 0.20 base, 0.30 strong trend, discrete levels, stoploss at 3*ATR (wider for 15m noise).
Timeframe: 15m (REQUIRED), HTF: 1h + 4h via mtf_data helper (call ONCE before loop).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_supertrend_rsi_4h_hma_1h_adx_v1"
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

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator.
    Returns: supertrend_values, supertrend_direction (1=long, -1=short)
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    hl2 = (high + low) / 2
    
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros(n)
    supertrend[:] = np.nan
    direction = np.zeros(n)  # 1 = long (price above ST), -1 = short (price below ST)
    
    # Initialize
    supertrend[period] = upper_band[period]
    direction[period] = -1
    
    for i in range(period + 1, n):
        if np.isnan(atr[i]):
            supertrend[i] = np.nan
            direction[i] = 0
            continue
            
        # If trend is currently up (price above supertrend)
        if direction[i - 1] == 1:
            # Lower band can only move up
            if lower_band[i] > supertrend[i - 1]:
                supertrend[i] = lower_band[i]
            else:
                supertrend[i] = supertrend[i - 1]
            
            # Check if trend flips
            if close[i] < supertrend[i]:
                supertrend[i] = upper_band[i]
                direction[i] = -1
            else:
                direction[i] = 1
        else:
            # Trend is down (price below supertrend)
            if upper_band[i] < supertrend[i - 1]:
                supertrend[i] = upper_band[i]
            else:
                supertrend[i] = supertrend[i - 1]
            
            # Check if trend flips
            if close[i] > supertrend[i]:
                supertrend[i] = lower_band[i]
                direction[i] = 1
            else:
                direction[i] = -1
    
    return supertrend, direction

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_sma(close, period):
    """Calculate SMA."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, lower

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    zscore = (close - sma) / (std + 1e-10)
    return zscore

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_1h = calculate_hma(df_1h['close'].values, 21)
    adx_1h, plus_di_1h, minus_di_1h = calculate_adx(
        df_1h['high'].values, 
        df_1h['low'].values, 
        df_1h['close'].values, 
        14
    )
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h)
    adx_1h_aligned = align_htf_to_ltf(prices, df_1h, adx_1h)
    plus_di_1h_aligned = align_htf_to_ltf(prices, df_1h, plus_di_1h)
    minus_di_1h_aligned = align_htf_to_ltf(prices, df_1h, minus_di_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    rsi_7 = calculate_rsi(close, 7)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    # Supertrend on 15m
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    
    # Bollinger Bands
    bb_upper, bb_lower = calculate_bollinger_bands(close, 20, 2.0)
    
    # Z-score for mean reversion
    zscore = calculate_zscore(close, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4) - CONSERVATIVE for 15m noise
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    SIZE_HALF = 0.12
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(400, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(ema_21[i]) or np.isnan(adx_1h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(supertrend[i]) or np.isnan(st_direction[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = primary trend bias (strongest filter)
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # 1h HMA = intermediate trend
        bull_trend_1h = close[i] > hma_1h_aligned[i]
        bear_trend_1h = close[i] < hma_1h_aligned[i]
        
        # 15m EMA alignment
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # Price vs SMA200
        above_sma200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_sma200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # === TREND STRENGTH / REGIME (from 1h ADX) ===
        trending_regime = adx_1h_aligned[i] > 20
        strong_trend = adx_1h_aligned[i] > 30
        ranging_regime = adx_1h_aligned[i] < 18
        
        # DI crossover on 1h
        di_bullish_1h = plus_di_1h_aligned[i] > minus_di_1h_aligned[i]
        di_bearish_1h = plus_di_1h_aligned[i] < minus_di_1h_aligned[i]
        
        # === SUPERTREND SIGNALS ===
        st_long = st_direction[i] == 1
        st_short = st_direction[i] == -1
        
        # Supertrend flip detection
        st_flip_long = st_direction[i] == 1 and (i > 0 and st_direction[i - 1] == -1)
        st_flip_short = st_direction[i] == -1 and (i > 0 and st_direction[i - 1] == 1)
        
        # === RSI CONDITIONS ===
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_neutral = 40 <= rsi[i] <= 60
        rsi_pullback_long = 40 <= rsi[i] <= 55  # pullback in uptrend
        rsi_pullback_short = 45 <= rsi[i] <= 60  # pullback in downtrend
        
        # === BOLLINGER BAND POSITION ===
        near_bb_lower = close[i] <= bb_lower[i] * 1.005 if not np.isnan(bb_lower[i]) else False
        near_bb_upper = close[i] >= bb_upper[i] * 0.995 if not np.isnan(bb_upper[i]) else False
        
        # === Z-SCORE ===
        zscore_oversold = zscore[i] < -1.5
        zscore_overbought = zscore[i] > 1.5
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS (multiple paths for more trades) ===
        
        # Path 1: Supertrend flip + HTF alignment (strongest signal)
        if st_flip_long and bull_trend_4h and bull_trend_1h:
            if strong_trend:
                new_signal = SIZE_STRONG
            else:
                new_signal = SIZE_BASE
        
        # Path 2: Supertrend long + trend continuation
        if st_long and bull_trend_4h:
            if ema_bullish and rsi_pullback_long:
                new_signal = SIZE_BASE
        
        # Path 3: RSI pullback in strong uptrend
        if bull_trend_4h and bull_trend_1h:
            if rsi_oversold and above_sma200:
                if di_bullish_1h:
                    new_signal = SIZE_HALF
        
        # Path 4: Mean reversion in ranging market
        if ranging_regime:
            if zscore_oversold and near_bb_lower:
                if above_sma200:  # only long mean reversion above SMA200
                    new_signal = SIZE_HALF
        
        # Path 5: EMA crossover + Supertrend confirmation
        if ema_bullish and st_long:
            if rsi[i] > 45 and rsi[i] < 65:
                if bull_trend_4h:
                    new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS (multiple paths for more trades) ===
        
        # Path 1: Supertrend flip + HTF alignment (strongest signal)
        if st_flip_short and bear_trend_4h and bear_trend_1h:
            if strong_trend:
                new_signal = -SIZE_STRONG
            else:
                new_signal = -SIZE_BASE
        
        # Path 2: Supertrend short + trend continuation
        if st_short and bear_trend_4h:
            if ema_bearish and rsi_pullback_short:
                new_signal = -SIZE_BASE
        
        # Path 3: RSI pullback in strong downtrend
        if bear_trend_4h and bear_trend_1h:
            if rsi_overbought and below_sma200:
                if di_bearish_1h:
                    new_signal = -SIZE_HALF
        
        # Path 4: Mean reversion in ranging market
        if ranging_regime:
            if zscore_overbought and near_bb_upper:
                if below_sma200:  # only short mean reversion below SMA200
                    new_signal = -SIZE_HALF
        
        # Path 5: EMA crossover + Supertrend confirmation
        if ema_bearish and st_short:
            if rsi[i] > 35 and rsi[i] < 55:
                if bear_trend_4h:
                    new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - Wider stops for 15m noise ===
        atr_mult = 3.0  # wider stop for 15m
        
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            
            current_stop = highest_close - atr_mult * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            current_stop = lowest_close + atr_mult * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - atr_mult * atr[i] if position_side > 0 else close[i] + atr_mult * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - atr_mult * atr[i] if position_side > 0 else close[i] + atr_mult * atr[i]
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