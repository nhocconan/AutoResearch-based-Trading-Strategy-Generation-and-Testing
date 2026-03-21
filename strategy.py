#!/usr/bin/env python3
"""
Experiment #313: 15m Regime-Adaptive with 4h HMA Trend + Choppiness Index + RSI Mean Reversion
Hypothesis: 15m timeframe is too noisy for pure trend-following (see exp#301 Sharpe=-4.374).
Solution: Use Choppiness Index to detect regime. In trending markets (CHOP<38.2), follow 4h HMA trend
with RSI pullback entries. In ranging markets (CHOP>61.8), use RSI mean reversion with SMA filter.
4h HMA provides macro bias to avoid counter-trend trades. ATR stops protect capital.
This combines regime detection + MTF trend + adaptive entry logic for 15m success.
Timeframe: 15m (required), HTF: 4h for trend bias.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_regime_chop_4h_hma_rsi_adaptive_atr_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = ranging market, CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        atr_sum = 0.0
        highest_high = high[i]
        lowest_low = low[i]
        
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
            highest_high = max(highest_high, high[j])
            lowest_low = min(lowest_low, low[j])
        
        price_range = highest_high - lowest_low
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    n = len(close)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    chop = calculate_choppiness(high, low, close, 14)
    sma200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.12
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(sma200[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # 4h macro trend bias
        trend_bullish = close[i] > hma_4h_aligned[i]
        trend_bearish = close[i] < hma_4h_aligned[i]
        
        # Regime detection via Choppiness Index
        trending_regime = chop[i] < 38.2
        ranging_regime = chop[i] > 61.8
        neutral_regime = not trending_regime and not ranging_regime
        
        # ADX trend strength
        strong_trend = adx[i] > 25
        weak_trend = adx[i] < 20
        
        # DI crossover
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = plus_di[i] < minus_di[i]
        
        # Price vs SMA200
        above_sma200 = close[i] > sma200[i]
        below_sma200 = close[i] < sma200[i]
        
        # RSI levels
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_neutral = 40 < rsi[i] < 60
        rsi_bullish = rsi[i] > 50
        rsi_bearish = rsi[i] < 50
        
        new_signal = 0.0
        
        # === TRENDING REGIME (CHOP < 38.2) ===
        if trending_regime:
            # Long: 4h bullish + ADX strong + DI bullish + RSI pullback (40-55)
            if trend_bullish and strong_trend and di_bullish and 40 < rsi[i] < 55:
                new_signal = SIZE_ENTRY
            # Long: 4h bullish + Above SMA200 + RSI > 50 + ADX > 20
            elif trend_bullish and above_sma200 and rsi_bullish and adx[i] > 20:
                new_signal = SIZE_ENTRY
            
            # Short: 4h bearish + ADX strong + DI bearish + RSI pullback (45-60)
            elif trend_bearish and strong_trend and di_bearish and 45 < rsi[i] < 60:
                new_signal = -SIZE_ENTRY
            # Short: 4h bearish + Below SMA200 + RSI < 50 + ADX > 20
            elif trend_bearish and below_sma200 and rsi_bearish and adx[i] > 20:
                new_signal = -SIZE_ENTRY
        
        # === RANGING REGIME (CHOP > 61.8) ===
        elif ranging_regime:
            # Long: RSI oversold + Above SMA200 (mean reversion with bias)
            if rsi_oversold and above_sma200 and weak_trend:
                new_signal = SIZE_ENTRY
            # Long: RSI < 30 + 4h bullish (strong mean reversion with trend bias)
            elif rsi[i] < 30 and trend_bullish:
                new_signal = SIZE_ENTRY
            
            # Short: RSI overbought + Below SMA200 (mean reversion with bias)
            elif rsi_overbought and below_sma200 and weak_trend:
                new_signal = -SIZE_ENTRY
            # Short: RSI > 70 + 4h bearish (strong mean reversion with trend bias)
            elif rsi[i] > 70 and trend_bearish:
                new_signal = -SIZE_ENTRY
        
        # === NEUTRAL REGIME (38.2 <= CHOP <= 61.8) ===
        else:
            # Conservative: only enter with strong 4h bias and RSI confirmation
            if trend_bullish and above_sma200 and rsi_bullish and adx[i] > 15:
                new_signal = SIZE_ENTRY * 0.8
            elif trend_bearish and below_sma200 and rsi_bearish and adx[i] > 15:
                new_signal = -SIZE_ENTRY * 0.8
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from highest)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from lowest)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals